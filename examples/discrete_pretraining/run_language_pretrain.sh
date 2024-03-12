torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:0 run_pretraining_discrete.py \
        --model_name_or_path jxie/sma-language-pretrained --dataset_name jxie/wiki_books --do_train --max_seq_length 1024 \
        --per_device_train_batch_size 64 --learning_rate 1e-4 --max_steps 1000000 --output_dir runs/sma-language-pretrained --optim adamw_torch \
        --warmup_steps 5000 --weight_decay 0.01 --save_steps 10000 --logging_steps 10 --overwrite_output_dir --ignore_mismatched_sizes \
        --input_key text --fp16 --train_split train --lr_scheduler_type cosine --tokenizer_name jxie/sma-language-pretrained \
        --ddp_find_unused_parameters false --masking_ratio 0.15 --dataloader_num_workers 4
