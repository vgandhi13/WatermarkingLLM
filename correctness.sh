#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=20G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH  --constraint=vram23
#SBATCH -t 48:00:00  # Job time limit
#SBATCH -o correctness-%j.out  # %j = job ID

source ./bin/activate
python batch_tests_kmeans_variation.py # batch_tests_kmeans1k.py # correctness.py