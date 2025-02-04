#!/bin/bash
#SBATCH --job-name=dw
#SBATCH --output=download_data_%j.log
#SBATCH --error=download_data_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --environment=mem_wm
#SBATCH --container-workdir=/users/sstapf/mem_wm/finetune 


srun --pty bash train_custom.sh 