#!/usr/bin/env bash

#SBATCH --nodes=# TODO
#SBATCH --ntasks-per-node=# TODO
#SBATCH --output=hello-hvd-%j.out
#SBATCH --error=hello-hvd-%j.err
#SBATCH --time=00:02:00
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:# TODO

#SBATCH --account=training2306

source /p/project/training2306/software_environment/activate.sh

# make sure all GPUs on a node are visible
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# use `srun` to start Horovod code
srun --cpu-bind=none python # TODO path/to/script.py
