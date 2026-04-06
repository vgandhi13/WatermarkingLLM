#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=200G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --constraint=vram48
#SBATCH -t 48:00:00  # Job time limit
#SBATCH -o logs/eval-%j.out  # %j = job ID

#----------------SETTABLE PARAMETERS------------------
export NUM_PROMPTS=20
export MODEL_NAME="mistralai/Mistral-7B-v0.1"
export MAX_TOKENS=256
export WATERMARK_KEY=42
export FASTTEXT_MODEL_PATH="/work/pi_adamoneill_umass_edu/Varun/cc.en.300.bin"
#----------------END SETTABLE PARAMETERS--------------

source venv/bin/activate
python3 evaluate.py
