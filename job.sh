#!/bin/bash

#SBATCH --job-name=Walk_LN_On3
#SBATCH --output=hparam_tuning_%j.out
#SBATCH --error=hparam_tuning_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
 #SBATCH --gpus-per-task=rtx8000:1
 #SBATCH --cpus-per-task=6
 #SBATCH --ntasks-per-node=1
#SBATCH --mem=70G

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Load any modules and activate your Python environment here
module load python/3.8  

Xvfb :1 -screen 0 1024x768x16 &
export DISPLAY=:1
export CUDA_VISIBLE_DEVICES=0 #for cuda device error

cd qflow-discrete/

# install or activate requirements
if ! [ -d "$SLURM_TMPDIR/env/" ]; then
    virtualenv $SLURM_TMPDIR/env/
    source $SLURM_TMPDIR/env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source $SLURM_TMPDIR/env/bin/activate
fi

# log into WandB
export WANDB_API_KEY="a2ff527595e001c6604deef5f2f3a8ed97c08407"
python -c "import wandb; wandb.login(key='$WANDB_API_KEY')"

# Run existing Python script in repo for tuning
python hparam_tuner.py