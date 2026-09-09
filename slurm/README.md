# Running this pipeline on PARAM Shakti (GPU)

## One-time setup
```bash
# on the cluster, from /scratch/$USER (jobs must run from here, not /home)
cd /scratch/$USER
git clone https://github.com/InfoSage05/BTP.git
cd BTP

# check what's actually available before editing the scripts below
module avail python
module avail pytorch
module avail cuda

# if there's no ready-made pytorch module, create your own env instead:
python -m venv ~/chf_env
source ~/chf_env/bin/activate
pip install torch pandas numpy scipy openpyxl CoolProp
```
**Edit the two `module load` lines in `slurm/pretrain_both.sh` and `slurm/run_stage.sh`**
to match whatever `module avail` actually showed you (or replace them with
`source ~/chf_env/bin/activate` if you went the venv route instead).

## Every script in scripts/chf_pipeline/ now auto-detects the GPU
No code changes needed per-run -- `device_utils.py` picks CUDA automatically
if present, falls back to CPU otherwise. You'll see
`[device_utils] Using GPU: <name>` printed at the start of every script's
output if it's actually using the GPU (check this in the log — if it says
"No CUDA GPU found" instead, something's wrong with the module/env setup,
not the code).

## Submitting jobs
```bash
# Stage 1: pretrain both architectures (the expensive part -- do this first)
sbatch slurm/pretrain_both.sh

# check status
squeue -u $USER

# once that finishes, run any other stage the same way:
sbatch slurm/run_stage.sh scripts/chf_pipeline/finetune_mlp.py
sbatch slurm/run_stage.sh scripts/chf_pipeline/finetune_transformer.py
sbatch slurm/run_stage.sh scripts/chf_pipeline/ensemble_mlp.py
sbatch slurm/run_stage.sh scripts/chf_pipeline/ensemble_transformer.py
```

## Getting results back to your own machine
```bash
# from your own computer, not the cluster:
scp -r username@paramshakti.iitkgp.ac.in:/scratch/username/BTP/data/processed ./data/
```
(add `-P 4422` if off-campus)

## If a job fails immediately
Check the `.log` file it wrote (e.g. `pretrain_12345.log`) first -- most
likely causes are: wrong module names in the `module load` lines, or the
project not actually being at `/scratch/$USER/BTP` yet.
