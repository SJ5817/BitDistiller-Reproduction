export MODEL_PATH='../models/MetaMath-7b-V1.0/'
export SAVE_PATH=$2
export MASTER_ADDR="localhost"
export MASTER_PORT="1321"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true  

deepspeed --num_gpus=4 train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $1 \
    --model_max_length 512 \
    --output_dir $SAVE_PATH \
    --logging_dir $3 \
    --num_train_epochs $4 \
    --bf16 True \
    --seed 42 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --evaluation_strategy "steps" \
    --eval_steps 10 \
    --load_best_model_at_end True \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 2 \
    --learning_rate 8e-6 \
    --lr_scheduler_type "constant" \
    --weight_decay 0. \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --deepspeed config/zero.json \
    --bits 2 \
    --quant_type int2-asym \
    --q_group_size 128 \
    --train_kd True \
    --kd_loss_type "cakld" \
    --max_train_samples 999999 \
    --clip ../../clip_cache/MetaMath-7b-V1.0/int2-g128.pt
