model_name_or_path="jxie/sma-physics-pretrained"

CUDA_VISIBLE_DEVICES=0 python3 run_tabular_classification.py --model_name_or_path $model_name_or_path --dataset_name jxie/higgs --do_train --do_eval \
    --num_classes 1 --per_device_train_batch_size 128 --per_device_eval_batch_size $bs --learning_rate 3e-5 --num_train_epochs 200 \
    --output_dir runs/$(basename $model_name_or_path)-cls-higgs-split_train_63k \
    --optim adamw_torch --weight_decay 1e-5 --logging_steps 100 --evaluation_strategy epoch \
    --overwrite_output_dir --ignore_mismatched_sizes --train_split train_63k --eval_split test_20k --lr_scheduler_type constant \
    --ddp_find_unused_parameters false --bce_loss --dataloader_num_workers 4 --save_strategy no  \
    --attention_dropout_prob 0.1 --drop_path_rate 0.1 --hidden_dropout_prob 0.1 --fp16 \