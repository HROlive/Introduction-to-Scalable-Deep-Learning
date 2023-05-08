#!/bin/bash
# bash -x for detailed shell debugging

# Example: $K$ = 8 workers to run
# Total number of tasks $T$ is equal to number of workers $T=K=8$

# If we take $g=4$ GPUs per each node,
# this then corresponds to $N=K/g=8/4=2$ compute nodes to allocate,

# Tasks per node is equal to number of local GPUs to be allocated, $t=g=4$

# For cpu cores, we have then $c=C/g=96/4 = 24$. Total number of available cores
# $C$ on Booster is  2 * 24 * 2 = 96 threads.



#SBATCH --ntasks=8
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --output=training-out.%j
#SBATCH --error=training-err.%j
#SBATCH --time=00:02:00
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:4

#SBATCH --account=training2306

source /p/project/training2306/software_environment/activate.sh

# make sure all GPUs on a node are visible
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# use srun to start Horovod code
srun --cpu-bind=none,v python train.py --batch_size=16
