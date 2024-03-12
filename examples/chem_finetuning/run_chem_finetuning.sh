optim_name=adamw_torch
wd=0.001
ws=50
bs=8
model_name_or_path="jxie/sma-chemistry-pretrained"
ad=0.1
dp=0.0

export NCCL_P2P_LEVEL=NVL
for split in 0 #1 2
do
	lr=1e-4
	for dataset_name in "esol" "bbbp" "freesolv"
	do
		torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:0 run_chem_finetuning.py --model_name_or_path $model_name_or_path \
			--dataset_name jxie/$dataset_name --do_train --do_eval \
			--per_device_train_batch_size $bs --per_device_eval_batch_size $bs --learning_rate $lr  --num_train_epochs 40  --output_dir runs/$(basename $model_name_or_path)-cls-$dataset_name-split_$split \
			--optim $optim_name --warmup_steps $ws --weight_decay $wd --logging_steps 100 \
			--evaluation_strategy epoch --overwrite_output_dir --ignore_mismatched_sizes --fp16 \
			--train_split "train_$split" --eval_split "test_$split" --lr_scheduler_type cosine \
			--ddp_find_unused_parameters false \
			--dataloader_num_workers 8 --save_strategy "no" --max_seq_length 512 --attention_dropout_prob $ad --drop_path_rate $dp
	done
	lr=3e-5
	for dataset_name in "lipop" "bace" "hiv"
	do
		torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:0 run_chem_finetuning.py --model_name_or_path $model_name_or_path \
			--dataset_name jxie/$dataset_name --do_train --do_eval \
			--per_device_train_batch_size $bs --per_device_eval_batch_size $bs --learning_rate $lr  --num_train_epochs 40  --output_dir runs/$(basename $model_name_or_path)-cls-$dataset_name-split_$split \
			--optim $optim_name --warmup_steps $ws --weight_decay $wd --logging_steps 100 \
			--evaluation_strategy epoch --overwrite_output_dir --ignore_mismatched_sizes --fp16 \
			--train_split "train_$split" --eval_split "test_$split" --lr_scheduler_type cosine \
			--ddp_find_unused_parameters false \
			--dataloader_num_workers 8 --save_strategy "no" --max_seq_length 512 --attention_dropout_prob $ad --drop_path_rate $dp
	done
done
echo "Done"
