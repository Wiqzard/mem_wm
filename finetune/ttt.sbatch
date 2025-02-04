#!/bin/bash

############################
# SLURM directives
############################
#SBATCH --job-name=dw                          # EDIT: job name
#SBATCH --output=outputs/slurm/train_%j.log    # stdout
#SBATCH --error=outputs/slurm/train_%j.err     # stderr
#SBATCH --time=10:00:00                        # EDIT: wall-clock time
#SBATCH --partition=normal                     # EDIT: partition/queue
#SBATCH --nodes=2                              # number of nodes
#SBATCH --ntasks-per-node=1                    # 1 task per node
#SBATCH --gpus-per-node=4                      # gpus per node
#SBATCH --cpus-per-task=288                    # cpu cores per task
#SBATCH --environment=mem_wm                   # EDIT: custom env (if applicable)
#SBATCH --container-workdir=/capstor/scratch/cscs/sstapf/mem_wm/finetune
#SBATCH --exclusive                            # if you need exclusive node usage

echo "START TIME: $(date)"

# auto-fail on any errors in this script
set -eo pipefail

# log every command (for debug)
set -x

############################
# Environment Setup
############################
# If you need to source an environment or activate conda, do it here. For example:
# source /path/to/some_setup_script
# conda activate stas-xxx

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Hugging Face and wandb
export HF_HOME=/capstor/store/cscs/swissai/a03/hf
export WANDB_API_KEY=11b574cdfa34332326c4a0a1ac8f7b06fb123637

# (Optional) If you want to redirect temp files to somewhere with more space
# export TMPDIR=/scratch

############################
# Accelerate Setup
############################
ACCELERATE_CONFIG_FILE="accelerate_config.yaml"  # or "accelerate.yaml" (adjust path as needed)

# From SLURM
NNODES=$SLURM_NNODES
GPUS_PER_NODE=4  # must match --gpus-per-node above
NUM_PROCESSES=$(( NNODES * GPUS_PER_NODE ))

# The first node (master) for rendezvous
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=6000   # or choose a free port

############################
# Training Arguments
############################
# (Same definitions as in train_custom_multi_node.sh, but here in one place)
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX1.5-5B-I2V"
    --model_name "cogvideox1.5-i2v-wm"
    --model_type "wm"
    --training_type "sft"
    --local_path /capstor/scratch/cscs/sstapf/mem_wm/outputs/transformer
)

OUTPUT_ARGS=(
    --output_dir "outputs"
    --report_to "wandb"
)

DATA_ARGS=(
    --data_root "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/train_set"
    --caption_column "prompts.txt"
    --image_column "images_gen_new_debug.txt"
    --video_column "videos_gen_new_debug.txt"
    --train_resolution "81x368x640"  # frames x height x width, frames should be 8N+1
)

TRAIN_ARGS=(
    --train_epochs 100
    --seed 42
    --batch_size 2
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"
)

SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory False
    --nccl_timeout 1800
)

CHECKPOINT_ARGS=(
    --checkpointing_steps 200
    --checkpointing_limit 2
    # --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"
)

VALIDATION_ARGS=(
    --do_validation true
    --validation_dir "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/validation_set"
    --validation_steps 200
    --validation_prompts "prompts.txt"
    --validation_images "images_100.txt"
    --validation_videos "videos_100.txt"
    --gen_fps 16
)

############################
# Final Program/Command
############################
# Combine all argument lists for 'train.py'
TRAINING_SCRIPT="train.py \
  ${MODEL_ARGS[@]} \
  ${OUTPUT_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${TRAIN_ARGS[@]} \
  ${SYSTEM_ARGS[@]} \
  ${CHECKPOINT_ARGS[@]} \
  ${VALIDATION_ARGS[@]} \
"

# In the template, LAUNCHER calls Accelerate via python -m accelerate.commands.launch
# We delay expansion of $SLURM_PROCID and hostname for each node by escaping $ signs in machine_rank and role lines
#LAUNCHER="python -u -m accelerate.commands.launch \
#    --rdzv_conf 'rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT' \
#    --config_file $ACCELERATE_CONFIG_FILE \
#    --num_processes $NUM_PROCESSES \
#    --num_machines $NNODES \
#    --main_process_ip $MASTER_ADDR \
#    --main_process_port $MASTER_PORT \
#    --machine_rank \$SLURM_PROCID \
#    --role \$(hostname -s|tr -dc '0-9'): --tee 3 \
#    $TRAINING_SCRIPT
#"
LAUNCHER="python -u -m accelerate.commands.launch \
    --rdzv_conf 'rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT' \
    --distributed_type deepspeed \
    --deepspeed_config_file configs/zero2.yaml \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --role \$(hostname -s|tr -dc '0-9'): --tee 3 \
    $TRAINING_SCRIPT
"

export CMD="$LAUNCHER"
echo "$CMD"

############################
# srun launch
############################
LOG_PATH="main_log.txt"

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
"

# Note: using 'bash -c' so that $SLURM_PROCID, $(hostname -s), etc. expand on each node
srun $SRUN_ARGS bash -c "$CMD" 2>&1 | tee -a "$LOG_PATH"

echo "END TIME: $(date)"
