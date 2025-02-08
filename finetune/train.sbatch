#!/bin/bash
#SBATCH --job-name=dw
#SBATCH --output=outputs/slurm/train_%j.log
#SBATCH --error=outputs/slurm/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --nodes=8
#SBATCH --environment=mem_wm
#SBATCH --container-workdir=/capstor/scratch/cscs/sstapf/mem_wm/finetune 
#SBATCH --account=a-a03


#srun bash train_custom.sh 
srun bash train_custom_multi_node.sh
#srun bash train_custom_multi_node5b.sh