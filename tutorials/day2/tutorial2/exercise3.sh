#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=exercise3-%j.out
#SBATCH --error=exercise3-%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:4
#SBATCH --account=training2306

source /p/project/training2306/software_environment/activate.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Use `srun` to start Horovod code
srun python exercise3.py
