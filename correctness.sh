#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --nodes=1 # Number of Nodes
#SBATCH --mem=100G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH  --constraint=vram48
#SBATCH -t 48:00:00  # Job time limit
#SBATCH -o correctness-%j.out  # %j = job ID

# Memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ./WatVenv/bin/activate
python eval/correctness.py # batch_tests_kmeans1k.py # correctness.py