export NCCL_P2P_LEVEL=NVL
model_name_or_path=jxie/sma-image-pretrained

torchrun --nproc_per_node 8 --rdzv_backend c10d --rdzv_endpoint localhost:0 run_image_classification.py \
	--model_name_or_path $model_name_or_path --dataset_name jxie/imagenet-100 --do_train --do_eval --num_classes 100 \
	--per_device_train_batch_size 8 --learning_rate 1e-4 --num_train_epochs 200 \
	--output_dir runs/imagenet100_cls_$(basename $model_name_or_path) --optim adamw_torch \
	--warmup_steps 10000 --weight_decay 0.01 --eval_steps 1000 --logging_steps 10 --evaluation_strategy steps \
	--overwrite_output_dir --ignore_mismatched_sizes --image_column_name image --fp16 --train_split train --eval_split validation --lr_scheduler_type cosine \
	--drop_path_rate 0.1 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 --bce_loss --ddp_find_unused_parameters false --image_height 192 --image_width 192 \
	--dataloader_num_workers 4 --save_strategy no --three_aug
# done
echo "Done"