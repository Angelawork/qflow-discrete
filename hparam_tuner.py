import argparse
import os
import wandb
import subprocess
import itertools
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import stable_baselines3 as sb3

def grid_search(hyperparameter_space):
    """Generate all combinations of hyperparameters."""
    keys = hyperparameter_space.keys()
    values = (hyperparameter_space[key] for key in keys)
    comb = [dict(zip(keys, c)) for c in itertools.product(*values)]
    return comb

def random_search(hyperparameter_space, num_samples):
    """Randomly sample combinations in hyperparameters' space."""
    sampled = []
    for _ in range(num_samples):
        sampled_comb = {}
        for key in hyperparameter_space:
            sampled_comb[key] = random.choice(hyperparameter_space[key])
        sampled.append(sampled_comb)
    return sampled

def run(set_up, hparams, env_id, seed):
    """Run training script with specified hparam."""
    hparams_str = "_".join([f"{k}{v}" for k, v in hparams.items()])
    run_name = f"_{hparams_str}"  
    command = ["python", "qflow_discrete.py", "--exp-name", run_name, "--env-id", env_id] 
    for param, value in set_up.items():
        if param != '--seed': 
            command.extend([param, str(value)])
    command.extend(["--seed", str(seed)])  

    for param, value in hparams.items():
        command.extend([param, str(value)])

    print(f"Executing command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    episodic_returns = [float(line.split('=')[-1].strip(']').strip('[')) for line in result.stdout.split('\n') if "episodic_return" in line]
    final_episodic_return = episodic_returns[-1] if episodic_returns else 0
    print(f"Final episodic return: {final_episodic_return}")

    return final_episodic_return

def hparam_tuning(set_up, hparam_space, seeds, environments,num_samples):
    results = []
    hparam_combinations = grid_search(hparam_space)
    #hparam_combinations = random_search(hparam_space, num_samples)
    for env_id in environments:
        for hparams in hparam_combinations:
            final_returns = []
            for seed in seeds:
                print(f"Env: {env_id}, Seed {seed}: Running training with hparams: {hparams}")
                final_return = run(set_up, hparams, env_id, seed)
                final_returns.append(final_return)

            avg_return = np.mean(final_returns)
            std_return = np.std(final_returns)
            results.append({"env_id": env_id, "hparams": hparams, "avg_return": avg_return, "std_return": std_return})
            print(f"Env: {env_id}, Hparams: {hparams}, Avg Return: {avg_return}, Std Return: {std_return}")
    return results

if __name__ == "__main__":
    set_up = {
        "--total-timesteps": 500000,
        "--track": True,
        "--wandb-project-name": "QFlow_Walker_LN_On",
        "--wandb-entity": "angela-h",
        "--autotune":True,
        "--capture-video":True,
    }

    hyperparameter_space = {
        "--target-entropy": [4,6],#0.5, 2, 4, 6, 8
        "--use-ln-q":[True],
        "--use-ln-policy":[True]
    }

    environments = ["Walker2d-v4"] #"HalfCheetah-v4",18;"Ant-v4","Walker2d-v4",18,"Hopper-v4",8;
    seeds = [42,128,456]
    results = hparam_tuning(set_up, hyperparameter_space, seeds, environments,num_samples=1)
