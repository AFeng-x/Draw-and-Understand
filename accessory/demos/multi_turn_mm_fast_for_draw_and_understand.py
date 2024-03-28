import random
import sys
import os
import json
sys.path.append(os.path.abspath(__file__).rsplit('/', 3)[0])

import argparse
import multiprocessing as mp
import numpy as np
from typing import List, Optional

import torch
import torch.distributed as dist

from fairscale.nn.model_parallel import initialize as fs_init


from accessory.util.misc import setup_for_distributed
from accessory.model.meta import MetaModel
from accessory.data.conversation import default_conversation
from PIL import Image, ImageDraw, ImageFont
from accessory.data.transform import get_transform
from accessory.data.conversation.transforms import vp_normalize, Transform_Visual_Prompts

from visual_prompt_gradio import gradio_worker, ModelFailure, Ready

def model_worker(
    rank: int, args: argparse.Namespace, barrier: mp.Barrier,
    request_queue: mp.Queue, response_queue: Optional[mp.Queue] = None,
) -> None:
    """
    The worker function that manipulates the GPU to run the inference.
    Exact n_gpu workers are started, with each one operating on a separate GPU.

    Args:
        rank (int): Distributed rank of the worker.
        args (argparse.Namespace): All command line arguments.
        barrier (multiprocessing.Barrier): A barrier used to delay the start
            of Web UI to be after the start of the model.
    """

    world_size = len(args.gpu_ids)
    gpu_id = args.gpu_ids[rank]
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
    )
    print(f"| distributed init on worker {rank}/{world_size}. "
          f"using gpu: {gpu_id}")
    fs_init.initialize_model_parallel(world_size)
    torch.cuda.set_device(gpu_id)

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    # set the print behavior.
    # setup_for_distributed(rank == 0)

    target_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16
    }[args.dtype]
    model = MetaModel.from_pretrained(args.pretrained_path, args.llama_type, args.llama_config, args.tokenizer_path,
                                      with_visual=True, max_seq_len=args.max_seq_len,
                                      mp_group=fs_init.get_model_parallel_group(),
                                      dtype=target_dtype, device="cpu" if args.quant else "cuda", )
    if args.quant:
        from accessory.util.quant import quantize
        print("Quantizing model to 4bit!")
        from transformers.utils.quantization_config import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig.from_dict(
            config_dict={
                "load_in_8bit": False,
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
            },
            return_unused_kwargs=False,
        )
        quantize(model, quantization_config)
        model.cuda()
    model.eval()
    # print(f"Model = {str(model)}")

    conv = default_conversation()
    conv_sep = conv.response_end_signal

    barrier.wait()

    while True:
        if response_queue is not None:
            response_queue.put(Ready())
        try:
            image, vp, chatbot, max_gen_len, temperature, top_p, img_transform = request_queue.get()

            ### transform image ###
            if image is not None:
                image = image.convert("RGB")
                width, height = image.size
                transform = get_transform(img_transform, getattr(model.llma, 'image_size', 448))
                image = transform(image).unsqueeze(0).cuda().to(target_dtype)
            else:
                print("No image input")
                image = None
            
            ### transform visual prompts ###
            if vp is not None:
                vp_input = vp
                for coor in vp_input:
                    if len(coor) == 4:
                        coor[2] = coor[2] - coor[0]
                        coor[3] = coor[3] - coor[1]
                print("The coordinates for [x0,y0,width,height] of visual prompts are :" + str(vp_input))
                sparse_vp_input = Transform_Visual_Prompts(vp_input, width, height).unsqueeze(0).cuda().to(target_dtype)
            else:
                sparse_vp_input = None

            conv.load_qas(chatbot)
            prompt = conv.get_prompt()

            with torch.cuda.amp.autocast(dtype=target_dtype, enabled=not args.quant):
                print(prompt)
                for stream_response in model.stream_generate(
                    prompt, sparse_vp_input, image,
                    max_gen_len, temperature, top_p
                ):
                    # print(stream_response)
                    end_pos = stream_response["text"].find(conv_sep)
                    if end_pos != -1:
                        stream_response["text"] = (
                            stream_response['text'][:end_pos].rstrip() + "\n"
                        )
                        stream_response["end_of_content"] = True

                    # keep a few characters if not end_of_content to avoid sending
                    # part of conv_sep before all of it is generated.
                    if not stream_response["end_of_content"]:
                        if len(stream_response["text"]) < len(conv_sep):
                            continue
                        stream_response["text"] = (
                            stream_response["text"][:-len(conv_sep)]
                        )

                    if response_queue is not None:
                        stream_response['input'] = prompt
                        response_queue.put(stream_response)

                    if stream_response["end_of_content"]:
                        break
                    
        except Exception as e:
            print(e)
            response_queue.put(ModelFailure())



if __name__ == "__main__":
    parser = argparse.ArgumentParser("LLaMA2-Accessory Chat Demo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--gpu_ids", type=int, nargs="+",
        help="A list of space-separated gpu ids to run the model on. "
             "The model will span across GPUs in tensor-parallel mode."
    )
    group.add_argument(
        "--n_gpus", type=int, default=1,
        help="Number of GPUs to run the model on. Equivalent to "
             "--gpu_ids 0 1 2 ... n-1"
    )
    parser.add_argument(
        "--pretrained_path", type=str, required=True, nargs="+",
        help="Path to the llama model checkpoints. A list of checkpoints is "
             "supported and will be merged from left to right.")
    parser.add_argument(
        "--llama_type", default=None, type=str, metavar="MODEL",
        help="LLaMA model type."
    )
    parser.add_argument(
        "--llama_config", type=str, default=None, nargs="*",
        help="Path to the llama model config json."
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=None,
        help="Path to the tokenizer.model file provided along with the LLaMA "
             "model."
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=4096,
        help="Max sequence length accepted by the pretrained model."
    )
    parser.add_argument(
        "--master_port", type=int, default=23560,
        help="A port used by the PyTorch distributed module to initialize."
    )
    parser.add_argument(
        "--master_addr", type=str, default="127.0.0.1",
        help="An address used by the PyTorch distributed module to initialize."
    )
    parser.add_argument(
        "--dtype", type=str, choices=["fp16", "bf16"], default="bf16",
        help="The dtype used for model weights and inference."
    )
    parser.add_argument(
        "--quant", action="store_true", default=False,
        help="enable quantization"
    )
    parser.add_argument(
        "--bind_all", action="store_true",
        help="Listen to all addresses on the host."
    )
    args = parser.parse_args()

    # check and setup gpu_ids to use
    if args.gpu_ids is None:
        if args.n_gpus is None:
            args.n_gpus = 1
        assert args.n_gpus > 0, (
            "The demo currently must run on a positive number of GPUs."
        )
        args.gpu_ids = list(range(args.n_gpus))

    # using the default "fork" method messes up some imported libs (e.g.,
    # pandas)
    mp.set_start_method("spawn")

    # setup the queues and start the model workers
    request_queues = []
    response_queue = mp.Queue()
    worker_processes = []
    barrier = mp.Barrier(len(args.gpu_ids) + 1)
    for rank, gpu_id in enumerate(args.gpu_ids):
        request_queue = mp.Queue()
        rank_response_queue = response_queue if rank == 0 else None
        process = mp.Process(
            target=model_worker,
            args=(rank, args, barrier, request_queue, rank_response_queue),
        )
        process.start()
        worker_processes.append(process)
        request_queues.append(request_queue)

    gradio_worker(request_queues, response_queue, args, barrier)