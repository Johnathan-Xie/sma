lr=5e-5
optim_name=adamw_torch
wd=0.01
ws=1000
bs=64
model_name_or_path="jxie/sma-biology-pretrained"
dataset_name="scop" # "stability" "fluorescence" 
ad=0.0
dp=0.1
export NCCL_P2P_LEVEL=NVL

torchrun --nproc_per_node 8 --rdzv_backend c10d --rdzv_endpoint localhost:0 run_bio_finetuning.py --model_name_or_path $model_name_or_path \
	--dataset_name jxie/$dataset_name --do_train --do_eval \
	--per_device_train_batch_size $bs --per_device_eval_batch_size $bs --learning_rate $lr  --num_train_epochs 30  --output_dir runs/$(basename $model_name_or_path)-cls-$dataset_name \
	--optim $optim_name --warmup_steps $ws --weight_decay $wd --logging_steps 100 \
	--evaluation_strategy epoch --overwrite_output_dir --ignore_mismatched_sizes --fp16 \
	--train_split "train" --eval_split "test" --lr_scheduler_type cosine \
   --ddp_find_unused_parameters false \
	--dataloader_num_workers 8 --save_strategy "no" --max_seq_length 1024 --attention_dropout_prob $ad --drop_path_rate $dp

dp=0.0
bs=16
dataset_name="stability"

torchrun --nproc_per_node 8 --rdzv_backend c10d --rdzv_endpoint localhost:0 run_bio_finetuning.py --model_name_or_path $model_name_or_path \
	--dataset_name jxie/$dataset_name --do_train --do_eval \
	--per_device_train_batch_size $bs --per_device_eval_batch_size $bs --learning_rate $lr  --num_train_epochs 30  --output_dir runs/$(basename $model_name_or_path)-cls-$dataset_name \
	--optim $optim_name --warmup_steps $ws --weight_decay $wd --logging_steps 100 \
	--evaluation_strategy epoch --overwrite_output_dir --ignore_mismatched_sizes --fp16 \
	--train_split "train" --eval_split "test" --lr_scheduler_type cosine \
    --ddp_find_unused_parameters false \
	--dataloader_num_workers 8 --save_strategy "no" --max_seq_length 1024 --attention_dropout_prob $ad --drop_path_rate $dp

bs=8
dataset_name="fluorescence"
torchrun --nproc_per_node 8 --rdzv_backend c10d --rdzv_endpoint localhost:0 run_bio_finetuning.py --model_name_or_path $model_name_or_path \
	--dataset_name jxie/$dataset_name --do_train --do_eval \
	--per_device_train_batch_size $bs --per_device_eval_batch_size $bs --learning_rate $lr  --num_train_epochs 30  --output_dir runs/$(basename $model_name_or_path)-cls-$dataset_name \
	--optim $optim_name --warmup_steps $ws --weight_decay $wd --logging_steps 100 \
	--evaluation_strategy epoch --overwrite_output_dir --ignore_mismatched_sizes --fp16 \
	--train_split "train" --eval_split "test" --lr_scheduler_type cosine \
    --ddp_find_unused_parameters false \
	--dataloader_num_workers 8 --save_strategy "no" --max_seq_length 1024 --attention_dropout_prob $ad --drop_path_rate $dp