
set -x -e

export NUM_GPUS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    # gr00t/experiment/launch_finetune.py \
    # --base_model_path  nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
    # --dataset_path /mnt/sata1/gr00t16/HF_HOME/lerobot/rss-hiverobots/restock_yaw \
    # --modality_config_path examples/g1/g1_XRtele_yaw/modality_config.py \
    # --embodiment_tag NEW_EMBODIMENT \
    # --num_gpus $NUM_GPUS \
    # --output_dir /mnt/sata1/gr00t16/g1_finetune/restock_yaw__bs32_lr1e4_shxep10000_g1 \
    # --max_steps 10000 \
    # --save_steps 1000 \
    # --save-total-limit 3 \
    # --warmup_ratio 0.05 \
    # --weight_decay 1e-5 \
    # --learning_rate 1e-4 \
    # --global_batch_size 32 \
    # --gradient_accumulation_steps 1 \
    # --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    # --num-shards-per-epoch 10000 \
    # --dataloader-num-workers 4 \
    # --shard-size 75 \
    # --use_wandb \

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path  nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
    --dataset_path /mnt/sata1/gr00t16/HF_HOME/lerobot/rss-hiverobots/restock_yaw_hand_IO \
    --modality_config_path examples/g1/g1_XRtele_yaw_hand_IO/modality_config.py \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus $NUM_GPUS \
    --output_dir /mnt/sata1/gr00t16/g1_finetune/restock_yaw_hand_IO__bs32_lr1e4_shxep10000_g1 \
    --max_steps 10000 \
    --save_steps 1000 \
    --save-total-limit 3 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --num-shards-per-epoch 10000 \
    --dataloader-num-workers 4 \
    --shard-size 75 \
    --use_wandb \


torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path  nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
    --dataset_path /mnt/sata1/gr00t16/HF_HOME/lerobot/rss-hiverobots/restock_yaw \
    --modality_config_path examples/g1/g1_XRtele_yaw/modality_config.py \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus $NUM_GPUS \
    --output_dir /mnt/sata1/gr00t16/g1_finetune/restock_yaw__bs16_lr1e4_shxep10000_g1 \
    --max_steps 10000 \
    --save_steps 1000 \
    --save-total-limit 3 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --num-shards-per-epoch 10000 \
    --dataloader-num-workers 4 \
    --shard-size 75 \
    --use_wandb \

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path  nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
    --dataset_path /mnt/sata1/gr00t16/HF_HOME/lerobot/rss-hiverobots/restock_yaw_hand_IO \
    --modality_config_path examples/g1/g1_XRtele_yaw_hand_IO/modality_config.py \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus $NUM_GPUS \
    --output_dir /mnt/sata1/gr00t16/g1_finetune/restock_yaw_hand_IO__bs16_lr1e4_shxep10000_g1 \
    --max_steps 10000 \
    --save_steps 1000 \
    --save-total-limit 3 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --num-shards-per-epoch 10000 \
    --dataloader-num-workers 4 \
    --shard-size 75 \
    --use_wandb \