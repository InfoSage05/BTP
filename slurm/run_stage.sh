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

# apps/python-package/python/3.10.13 module is broken on this cluster (its
# PATH entry doesn't exist on disk) -- use the conda env instead:
# conda create -n chf_env python=3.10
source /home/apps/anaconda3/etc/profile.d/conda.sh
conda activate chf_env
module load compiler/cuda/12.4

python -u "$1"
