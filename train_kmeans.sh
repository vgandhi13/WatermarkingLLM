#!/bin/bash
#SBATCH -c 1  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH  --constraint=vram23
#SBATCH -t 05:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

source ./bin/activate
python kmeans_n_variation.py