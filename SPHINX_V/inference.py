from SPHINX_V import SPHINX_V_Model
from PIL import Image
import os
import torch
import torch.distributed as dist
import pdb

def main() -> None:
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ["RANK"])
    dist.init_process_group(
        world_size=world_size, rank=rank,
        backend="nccl", init_method=f"env://",
    )
    torch.cuda.set_device(rank)

    # torch.cuda.set_device(0)

    model = SPHINX_V_Model.from_pretrained(
        pretrained_path="/data1/weifeng/eccv2024/LLaMA2-Accessory/accessory/output/stage2_final/epoch0", 
        llama_type="llama_ens5_vp",
        llama_config="../accessory/checkpoints/llama-2-13b/params.json",
        with_visual=True,
        mp_group=None # dist.new_group(ranks=list(range(world_size)))
    )
    # You may also, say, launch 4 processes and make [0,1] and [2,3] ranks to form mp groups, respectively.

    # it's important to make sure that ranks within the same 
    # model parallel group should always receive the same input simultaneously
    image = Image.open("examples/1.jpg")
    # Please provide a detailed description of each marked region in the image.
    qas = [["Please analyze the relationship between all marked regions in the image.", None]]

    vps = [[49,239,245,421],[724,57,130,179],[386,274,452,348]]

    response = model.generate_response(qas, vps, image, max_gen_len=1024, temperature=0.9, top_p=0.5, seed=0)
    print(response)
    pdb.set_trace()
    # if you wanna continue
    qas[-1][-1] = response
    qas.append(["Can you tell what kind of antelope is in <Region 1> and what it is doing", None])
    response2 = model.generate_response(qas, vps, image, max_gen_len=1024, temperature=0.9, top_p=0.5, seed=0)
    print(response2)


if __name__ == "__main__":
    # launch this script with `torchrun --master_port=1112 --nproc_per_node=1 inference.py`
    main()