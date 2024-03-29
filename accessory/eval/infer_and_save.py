from SPHINX_V import SPHINX_V_Model
from PIL import Image
import os
import torch
import torch.distributed as dist
import json
import argparse
import random
from tqdm import tqdm
import pdb

def collate_fn(batches):
    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]
    image_path = [_['image_path'] for _ in batches]
    raw_question = [_['question_raw'] for _ in batches]

    input_sparse_vp = torch.cat([_['sparse_vp'] for _ in batches])
    input_image = torch.cat([_['image'] for _ in batches])

    return input_image, input_sparse_vp, question_ids, questions, annotations, image_path, raw_question

def get_qa_list(json_path):
    json_path = json_path.replace("eval", ".")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def vp_normalize(in_p, pad_x, pad_y, width, height):
    if len(in_p) == 2:
        x0, y0 = in_p
        x0 = x0 + pad_x
        y0 = y0 + pad_y
        sx0 = round(x0 / width, 3)
        sy0 = round(y0 / height,3)
        return [sx0, sy0, -1, -1]
    elif len(in_p) == 4:
        x0, y0, w, h = in_p
        x0 = x0 + pad_x
        y0 = y0 + pad_y
        sx0 = round(x0 / width, 3)
        sy0 = round(y0 / height, 3)
        sx1 = round((x0 + w) / width, 3)
        sy1 = round((y0 + h) / height, 3)
        return [sx0, sy0, sx1, sy1]

def Transform_Visual_Prompts(vp, width, height):
    if height > width:
        pad_x0 = int((height - width) / 2)
        pad_y0 = 0
        width = height
    else:
        pad_x0 = 0
        pad_y0 = int((width - height) / 2)
        height = width

    vp_length = len(vp)
    if len(vp[0]) == 2:
        label = "point"
    elif len(vp[0]) == 4:
        label = "box"
    else:
        assert False, "vp length error"

    for i, item in enumerate(vp):
        norm_item = vp_normalize(item,pad_x0,pad_y0,width,height)
        vp[i] = norm_item
    if vp_length > 10:
        vp = vp[:10]
    else:
        if label == "point":
            while vp_length < 10:
                vp.append([0, 0, -1, -1])
                vp_length += 1
        else:
            while vp_length < 10:
                vp.append([0, 0, 0, 0])
                vp_length += 1

    sparse_vp_input = torch.tensor(vp)

    return sparse_vp_input


def main(args):
    torch.cuda.set_device(0)
    # define the model
    model = SPHINX_V_Model.from_pretrained(
        pretrained_path=args.pretrained_path[0], 
        llama_type=args.llama_type,
        llama_config=args.llama_config[0],
        with_visual=True,
        mp_group=None
    )

    print(f"##### load pretrained from {args.pretrained_path} #####")

    with open('./vp_annotation_config.json', 'r') as f:
        ds_collections = json.loads(f.read())

    if args.dataset[0] == 'all':
        dataset_names = ds_collections.keys()
    else:
        dataset_names = args.dataset

    for ds in dataset_names:
        base_result_dir = f'results/{args.pretrained_path[0].split("ckpts")[-1].replace("/", "_")}'
        if not os.path.exists(base_result_dir):
            os.mkdir(base_result_dir)
        result_json_path = f"{base_result_dir}/{ds}_results.json"
        log_dir = f'{base_result_dir}/results.txt'
        max_gen_len = ds_collections[ds]['max_new_tokens']
        outputs = []
        if os.path.exists(log_dir):
            with open(log_dir, 'r') as f:
                pre_log = f.read()
            if ds in pre_log:
                print(f'##### Dataset: {ds} is tested, skip here. #####')
                # continue # FIXME
        qa_list = get_qa_list(ds_collections[ds]['test'])
        for qa in tqdm(qa_list, desc=f"##### {ds} evaluating #####"):
            image = Image.open(f"{args.img_root}{qa['image_name']}")
            qas = [[qa['question'], None]]
            if 'bbox' in qa:
                vps = qa['bbox']
            else:
                vps = qa['points']
            response = model.generate_response(qas, vps, image, max_gen_len=max_gen_len, temperature=args.temperature, top_p=args.top_p, seed=0)
            item = {
                "image_name":  qa['image_name'],
                "question_id": qa['question_id'],
                "question": qa['question'],
                "prediction": response,
                "answer": qa['gt_answers']
            }
            outputs.append(item)
        with open(result_json_path, "w") as f:
            json.dump(outputs, f)
        print(f"##### {result_json_path} saved successfully! #####")


        


if __name__ == '__main__':
    def get_args_parser():
        parser = argparse.ArgumentParser('Single-turn (conversation) demo', add_help=False)
        # Model parameters
        parser.add_argument('--llama_type', default='llama_ens5_vp', type=str, metavar='MODEL',
                            help='type of llama')
        parser.add_argument('--llama_config', default="path to llama_config", type=str, nargs="+",
                            help='Path to llama model config')
        parser.add_argument('--tokenizer_path', type=str, default="path to tokenizer_path",
                            help='path to tokenizer.model')
        parser.add_argument('--img_root', type=str, default="path to img_root_dir",
                            help='path to image')
        parser.add_argument('--pretrained_path', default="path to pretrained_model_path", type=str, nargs="+",
                            help='directory containing pre-trained checkpoints')
        parser.add_argument('--device', default='cuda',
                            help='device for inference')
        parser.add_argument('--model_parallel_size', default=1, type=int)
        parser.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
        parser.add_argument('--batch_size', default=2, type=int)
        parser.add_argument('--num_workers', default=4, type=int)
        parser.add_argument('--local_rank', default=-1, type=int)
        parser.add_argument('--seed', default=1, type=int)
        parser.add_argument('--dist_on_itp', action='store_true')
        parser.add_argument('--dist_url', default='env://',
                            help='url used to set up distributed training')
        parser.add_argument('--dataset', default="dataset to evaluate", type=str, nargs="+")
        parser.add_argument("--max_seq_length", type=int, default=4096)
        parser.add_argument("--master_port", type=int, default=None)
        parser.add_argument("--master_addr", type=str, default="127.0.0.1")
        parser.add_argument("--temperature", type=float, default=0.1)
        parser.add_argument("--top_p", type=float, default=0.75)

        return parser

    args = get_args_parser().parse_args()

    main(args)
