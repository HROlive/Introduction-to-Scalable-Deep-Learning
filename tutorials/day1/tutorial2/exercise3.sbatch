#!/usr/bin/env bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu
#SBATCH --time=00:05:00
#SBATCH --partition=dc-gpu
#SBATCH --account=training2306

source /p/project/training2306/software_environment/activate.sh

srun python exercise3.py
