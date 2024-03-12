torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:0 run_image_pretraining.py --model_name_or_path jxie/sma-image-pretrained \
        --dataset_name jxie/imagenet-100 --do_train \
        --per_device_train_batch_size 32 --learning_rate 1e-4  --num_train_epochs 200  --output_dir runs/sma-image-pretrained \
        --optim adamw_torch --warmup_steps 5000 --weight_decay 0.01 --save_steps 10000 --logging_steps 10 \
        --overwrite_output_dir --ignore_mismatched_sizes  --image_column_name image --fp16 \
        --train_split "train" --lr_scheduler_type cosine \
        --resize_method random_resized_crop --image_height 192 --image_width 192 