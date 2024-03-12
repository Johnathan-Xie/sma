# Note that the loading method will not copy the weights and instead give a scratch initialization
torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:0 run_pretraining_discrete.py --model_name_or_path jxie/sma-biology-pretrained \
        --do_train --max_seq_length 1024 --per_device_train_batch_size 64 --learning_rate 3e-4 --num_train_epochs 10 \
        --output_dir runs/sma-biology-pretrained --optim adamw_torch --warmup_steps 5000 --weight_decay 0.01 \
        --save_steps 100000 --logging_steps 10 --overwrite_output_dir --ignore_mismatched_sizes --input_key text \
        --dataset_name jxie/pfam --train_split train --min_seq_length 50 --lr_scheduler_type cosine \
        --tokenizer_name jxie/sma-biology-pretrained --save_total_limit 5 \
        --masking_ratio 0.15 --fp16 --no_pretokenization --ddp_timeout 6000 --dataloader_num_workers 4 --ddp_find_unused_parameters false