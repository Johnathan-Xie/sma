torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:0 run_pretraining_discrete.py --model_name_or_path jxie/sma-chemistry-pretrained \
        --dataset_name jxie/guacamol --do_train --tokenizer_name jxie/sma-chemistry-pretrained \
        --per_device_train_batch_size 32 --learning_rate 1e-4  --num_train_epochs 30  --output_dir runs/sma-chemistry-pretrained \
        --optim adamw_torch --warmup_steps 10000 --weight_decay 0.01 --save_steps 10000 --logging_steps 10 \
        --overwrite_output_dir --ignore_mismatched_sizes --fp16 --train_split "train" --lr_scheduler_type cosine \
        --ddp_find_unused_parameters false --dataloader_num_workers 8 --max_seq_length 512 --masking_ratio 0.3 --merge_method sep \
        --ddp_timeout 3000 --pad_to_max_length --overwrite_cache