### stage 2 fine-tuning training script

GPU=0,1,2,3,4,5,6,7
pretrained_path=checkpoints/sphinx-v/stage1
resume_path=checkpoints/sphinx-v/stage1

pretrained_type=consolidated
llama_config="checkpoints/llama-2-13b/params.json"
tokenizer_path="checkpoints/tokenizer/tokenizer.model"
data_config=configs/data/vp_finetune.yaml

data_parallel=sdp
model_parallel=2
exp_name=finetune/SPHINX-V_stage2
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"

### If need resume to train, please add: --resume "$resume_path"

CUDA_VISIBLE_DEVICES=${GPU} torchrun --nproc_per_node=8 main.py \
--output_dir output/"$exp_name" --epochs 1 --warmup_epochs 0.03 \
--batch_size 4 --accum_iter 4 --num_workers 4 \
--max_words 3072 \
--vpe_freeze \
--lr 0.00001 --min_lr 0 --clip_grad 8 --weight_decay 0 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" --checkpointing \
--llama_type llama_ens5_vp --llama_config $llama_config --tokenizer_path "$tokenizer_path" \
--pretrained_path "$pretrained_path" --pretrained_type="$pretrained_type" \
--data_config $data_config --dialog --save_iteration_interval 5000 \
--image_transform padded_resize \
2>&1 | tee -a output/"$exp_name"/output.log

echo "exp name: $exp_name"
