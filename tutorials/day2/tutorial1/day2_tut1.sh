#!/bin/bash
# `bash -x` for detailed Shell debugging

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --partition=dc-gpu

#SBATCH --output=day2-tut1-%j.out
#SBATCH --error=day2-tut1-%j.err

#SBATCH --account=training2306


source /p/project/training2306/software_environment/activate.sh


srun python -u Day2_Tutorial1_DL_Basics_Recap.py
