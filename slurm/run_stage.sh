#!/bin/bash
# Generic runner for any Stage 2/3/4 script -- pass the script name as an argument.
# Usage: sbatch slurm/run_stage.sh scripts/chf_pipeline/finetune_transformer.py
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32000
#SBATCH -t 06:00:00
#SBATCH -J chf_stage
#SBATCH -o stage_%j.log

cd /scratch/$USER/BTP || { echo "Project not found at /scratch/$USER/BTP -- git clone it first"; exit 1; }

module load python/3.10
module load pytorch/2.x-cuda

python -u "$1"
