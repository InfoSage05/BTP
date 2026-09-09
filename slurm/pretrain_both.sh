#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32000
#SBATCH -t 04:00:00
#SBATCH -J chf_pretrain
#SBATCH -o pretrain_%j.log

# Run from /scratch/$USER/BTP (per PARAM Shakti's rule: jobs must submit from /scratch)
cd /scratch/$USER/BTP || { echo "Project not found at /scratch/$USER/BTP -- git clone it first"; exit 1; }

# ---- adjust these two lines to whatever `module avail` actually shows ----
module load python/3.10
module load pytorch/2.x-cuda   # or e.g. cuda/11.8 + a conda env with torch installed separately
# ---------------------------------------------------------------------------

python -u scripts/chf_pipeline/pretrain.py mlp
python -u scripts/chf_pipeline/pretrain.py transformer
