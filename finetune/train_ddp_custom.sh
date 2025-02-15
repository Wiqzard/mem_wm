#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    #--model_path "THUDM/CogVideoX1.5-5B-I2V"
    --model_path  "THUDM/CogVideoX-2b"
    #--model_name "cogvideox1.5-i2v-wm"  # ["cogvideox-i2v"]
    --model_name  "cogvideox-i2v-wm"
    #--model_type "i2v"
    --model_type "wm"
    #--training_type "lora"
    --training_type "sft"
    #--local_path /home/ss24m050/Documents/CogVideo/outputs/transformer
    #--local_path /home/ss24m050/Documents/CogVideo/outputs/transformer_2b_iv
    #--local_path "/home/ss24m050/Documents/CogVideo/outputs/transformer_2b_iv_grp_2"
    #--local_path "/home/ss24m050/Documents/CogVideo/outputs/transformer_2b_iv_grp_2"
    --local_path "/home/ss24m050/Documents/CogVideo/ckpts/cogvideo-2b"
    #--encoder_path 
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "outputs"
    --report_to "wandb"
  
)

# Data Configuration
DATA_ARGS=(
    --data_root "/home/ss24m050/Documents/CogVideo/data/processed_gf_3/train_set" #"/home/ss24m050/Documents/CogVideo/data_test/post"
    --caption_column "prompts.txt"
    --image_column "images_train_processed_gf.txt" #"images.txt"
    --video_column "videos_train_processed_gf.txt" #"videos.txt"
    --train_resolution "49x352x640"  # (frames x height x width), frames should be 8N+1
    --encode_online 1
    #--train_resolution "49x352x640"  # (frames x height x width), frames should be 8N+1
    #--train_resolution "49x36x640"  # (frames x height x width), frames should be 8N+1
    #--train_resolution "81x360x640"  # (frames x height x width), frames should be 8N+1
    # --image_column "images.txt"  # comment this line will use first frame of video as image conditioning
    #--train_resolution "81x768x1360"  # (frames x height x width), frames should be 8N+1
    #--train_resolution "49x480x720"  # (frames x height x width), frames should be 8N+1
    #--train_resolution "49x240x360"  # (frames x height x width), frames should be 8N+1

)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 100 # number of training epochs
    --seed 42 # random seed
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 50 # save checkpoint every x steps
    --checkpointing_limit 2 # maximum number of checkpoints to keep, after which the oldest one is deleted
  #  --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true #true #false  # ["true", "false"]
    --validation_dir "/home/ss24m050/Documents/CogVideo/data/processed_gf_3/val_set" #"/home/ss24m050/Documents/CogVideo/data_test/post" #"/home/ss24m050/Documents/CogVideo/data/data_269"
    --validation_steps 50  # should be multiple of checkpointing_steps
    --validation_prompts "prompts_val_processed_gf.txt"
    --validation_images "images_val_processed_gf.txt"
    --validation_videos "videos_val_processed_gf.txt"
    --gen_fps 16
)

# Combine all arguments and launch training
#accelerate launch train_online.py \
accelerate launch --debug train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"
