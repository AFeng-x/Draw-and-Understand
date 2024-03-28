pretrained_path="accessory/checkpoints/sphinx-v/stage2"
llama_config="accessory/checkpoints/llama-2-13b/params.json"
tokenizer_path=".accessory/checkpoints/tokenizer/tokenizer.model"

CUDA_VISIBLE_DEVICES=6,7 python multi_turn_mm_fast_for_draw_and_understand.py \
  --n_gpus=2 --master_port 23353 \
  --llama_type llama_ens5_vp \
  --llama_config ${llama_config} \
  --tokenizer_path ${tokenizer_path} \
  --pretrained_path ${pretrained_path} \
