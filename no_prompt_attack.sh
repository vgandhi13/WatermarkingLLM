#!/bin/bash
#SBATCH -c 1  # Number of Cores per Task
#SBATCH --mem=30G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH  --constraint=vram32
#SBATCH -t 48:00:00  # Job time limit
#SBATCH -o no_prompt_attack_k_variation-%j.out  # %j = job ID

source ./bin/activate
python eval/no_prompt_attack_k_variation.py