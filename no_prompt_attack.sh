#!/bin/bash
#SBATCH -c 1  # Number of Cores per Task
#SBATCH --mem=350G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH  --constraint=vram48
#SBATCH -t 48:00:00  # Job time limit
#SBATCH -o no_prompt_attack_k_variation-%j.out  # %j = job ID

source ./WatVenv/bin/activate
python eval/no_prompt_attack_k_variation.py