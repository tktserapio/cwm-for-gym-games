#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --job-name=cwm_job
#SBATCH --output=cwm_logs/cwm_job_%j.out
#SBATCH --error=cwm_logs/cwm_job_%j.err

# Load modules
module load python/3.11

# Navigate and activate venv
cd ~/cwm-for-gym-games
source .venv/bin/activate
python main.py