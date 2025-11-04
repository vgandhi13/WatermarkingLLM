#!/bin/bash
#SBATCH -c 1  # Number of Cores per Task
#SBATCH --mem=80G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH  --constraint=vram32
#SBATCH -t 48:00:00  # Job time limit
#SBATCH -o randomcode-%j.out  # %j = job ID

source ./WatVenv/bin/activate
python random_linear_code.py