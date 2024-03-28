### stage 1 pre-train training script

GPU=0,1,2,3,4,5,6,7
pretrained_path=checkpoints/SPHINX-v2-1k
resume_path=checkpoints/SPHINX-v2-1k

pretrained_type=consolidated
llama_config="checkpoints/llama-2-13b/params.json"
tokenizer_path="checkpoints/tokenizer/tokenizer.model"
data_config=configs/data/vp_pretrain.yaml

data_parallel=sdp
model_parallel=2
exp_name=pretrain/SPHINX-V_stage1
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"

### If need resume to train, please add: --resume "$resume_path"

CUDA_VISIBLE_DEVICES=${GPU} torchrun --nproc_per_node=4 main.py \
--output_dir output/"$exp_name" --epochs 1 --warmup_epochs 0.03 \
--batch_size 24 --accum_iter 4 --num_workers 4 \
--max_words 2048 \
--llm_freeze \
--lr 0.00004 --min_lr 0 --clip_grad 8 --weight_decay 0 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" --checkpointing \
--llama_type llama_ens5_vp --llama_config $llama_config --tokenizer_path "$tokenizer_path" \
--pretrained_path "$pretrained_path" --pretrained_type="$pretrained_type" \
--data_config $data_config --dialog --save_iteration_interval 5000 \
--image_transform padded_resize \
2>&1 | tee -a output/"$exp_name"/output.log

echo "exp name: $exp_name"
