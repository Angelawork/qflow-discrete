#!/bin/bash

#SBATCH --job-name=SAC
#SBATCH --output=./Err_log/SAC_%j.out
#SBATCH --error=./Err_log/SAC_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
 #SBATCH --gpus-per-task=rtx8000:1
 #SBATCH --cpus-per-task=6
 #SBATCH --ntasks-per-node=1
#SBATCH --mem=70G

echo "Date:     $(date)"
echo "Hostname: $(hostname)"

module load python/3.8

cd cleanrl/

if ! [ -d "$SLURM_TMPDIR/env/" ]; then
    virtualenv $SLURM_TMPDIR/env/
    source $SLURM_TMPDIR/env/bin/activate
    pip install --upgrade pip
    pip install -r ./requirements/requirements-mujoco.txt
    pip install --upgrade typing_extensions
    pip install torch torchvision torchaudio --force-reinstall  --extra-index-url https://download.pytorch.org/whl/cu116
else
    source $SLURM_TMPDIR/env/bin/activate
fi

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY="a2ff527595e001c6604deef5f2f3a8ed97c08407"
python -c "import wandb; wandb.login(key='$WANDB_API_KEY')"

# default value for seed, (42,128,456)

python ./cleanrl/sac_continuous_action.py --env-id HumanoidStandup-v4 \
    --seed 42 \
    --target_entropy 17.0 \
    --total-timesteps 1500000 \
    --track \
    --wandb-project-name "Cleanrl_SAC"