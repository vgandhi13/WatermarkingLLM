#!/bin/bash
#SBATCH -c 1  # Number of Cores per Task
#SBATCH --mem=100G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH  --constraint=vram48
#SBATCH -t 48:00:00  # Job time limit
#SBATCH -o translate-%j.out  # %j = job ID
source ./WatVenv/bin/activate
python eval/translation.py