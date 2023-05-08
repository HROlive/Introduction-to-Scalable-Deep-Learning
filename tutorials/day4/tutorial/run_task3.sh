#!/bin/bash -x
#SBATCH --account=training2306
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:4
#SBATCH --partition=dc-gpu

if [ "$#" -lt 1 ]; then
    echo Missing batch size argument!
    exit 1
fi

source /p/project/training2306/software_environment/activate.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES="0,1,2,3"
BATCH_SIZE=$1
srun --cpu-bind=none,v --accel-bind=gn  python -u train.py  --epochs 150 --batch_size $BATCH_SIZE --learning_rate 0.1 --enable_lr_rescaling --lr_scheduler=step_wise_decay_with_warmup --result_file "task3_bs$BATCH_SIZE.csv"
