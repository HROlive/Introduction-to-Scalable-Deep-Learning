#!/bin/bash -x
#SBATCH --account=training2306
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:4
#SBATCH --partition=dc-gpu
source /p/project/training2306/software_environment/activate.sh
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES="0,1,2,3"
srun --cpu-bind=none,v --accel-bind=gn  python -u dcgan_multi_node.py --epochs=200 --batch_size=32
