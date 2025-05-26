#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=100G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH  --constraint=a100-80g
#SBATCH -t 48:00:00  # Job time limit
#SBATCH -o paraphase-%j.out  # %j = job ID

source ./bin/activate
python eval/paraphrase.py