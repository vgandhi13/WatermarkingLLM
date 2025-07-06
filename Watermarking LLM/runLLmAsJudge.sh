#!/bin/bash
#SBATCH -c 1  # Number of Cores per Task
#SBATCH --mem=100G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH  --constraint=a100-80g
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o llmJudgey-%j.out  # %j = job ID
#module load uri/main nodejs/14.17.6-GCCcore-11.2.0
source ./bin/activate
python eval/llmAsJudge.py
