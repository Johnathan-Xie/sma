model_name_or_path=jxie/sma-language-pretrained

for task_name in rte mrpc stsb cola sst2 qnli qqp mnli
do
        CUDA_VISIBLE_DEVICES=0 python run_glue.py  --model_name_or_path $model_name_or_path --task_name $task_name  --do_train  --do_eval \
        --max_seq_length 2048  --per_device_train_batch_size 32  --learning_rate 2e-5  --num_train_epochs 5  --output_dir runs/$task_name-$(basename $model_name_or_path) \
        --optim adamw_torch --warmup_steps 200 --weight_decay 0.01 --logging_steps 10 \
        --evaluation_strategy epoch --overwrite_output_dir --ignore_mismatched_sizes --fp16 --save_strategy no
done