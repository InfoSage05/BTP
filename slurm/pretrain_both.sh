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

# The apps/python-package/python/3.10.13 module is broken on this cluster
# (its PATH entry, /apps/python/3.10.13/bin, doesn't exist on disk) -- use
# the conda env built directly with `conda create -n chf_env python=3.10`
# instead, which sidesteps that entirely.
source /home/apps/anaconda3/etc/profile.d/conda.sh
conda activate chf_env
module load compiler/cuda/12.4

echo "python: $(which python) ($(python --version))"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

python -u scripts/chf_pipeline/pretrain.py mlp
python -u scripts/chf_pipeline/pretrain.py transformer
