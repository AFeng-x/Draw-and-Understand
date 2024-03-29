GPU=0
pretrained_path="/data1/weifeng/eccv2024/LLaMA2-Accessory/accessory/output/stage2_final/epoch0"
llama_config="/data1/weifeng/eccv2024/LLaMA2-Accessory/accessory/checkpoints/llama-2-13b/params.json"
tokenizer_path="/data1/weifeng/eccv2024/LLaMA2-Accessory/accessory/checkpoints/tokenizer/tokenizer.model"


CUDA_VISIBLE_DEVICES=${GPU} PYTHONWARNINGS="ignore" python -W ignore infer_and_save.py \
--dataset Osprey_80_detail_caption \
--llama_type llama_ens5_vp \
--llama_config ${llama_config} \
--tokenizer_path ${tokenizer_path} \
--pretrained_path ${pretrained_path} \
--img_root "/data1/coco/allimgs2017/" \
--batch_size 2 \
--model_parallel_size 1


