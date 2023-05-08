#!/bin/bash
#SBATCH --nodes=1
#SBATCH -A training2306
#SBATCH --partition=dc-gpu
#SBATCH --gres gpu
#SBATCH --time=00:05:00

echo `hostname` > output.txt
