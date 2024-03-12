torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:0 run_tabular_pretraining.py --model_name_or_path jxie/sma-physics-pretrained \
        --dataset_name jxie/higgs --do_train --per_device_train_batch_size 64 --learning_rate 1e-4 --num_train_epochs 50 --output_dir runs/sma-physics-pretrained \
        --optim adamw_torch --warmup_steps 10000 --weight_decay 0.01 --save_steps 100000 --logging_steps 100 --overwrite_output_dir --ignore_mismatched_sizes \
        --train_split train --lr_scheduler_type cosine --ddp_find_unused_parameters false --dataloader_num_workers 8 --fp16 --masking_ratio 0.2 --masking_schedule_length_ratio 0.25