# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from qflow_model import GFN
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="advantage-diffusion",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default='swish',
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Hopper-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning")
    parser.add_argument("--q-lr", type=float, default=5e-4,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--q-steps", type=int, default=1,
        help="Number of Q steps per env step")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--alpha", type=float, default=1.0,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--target-entropy", type=float, default=2.0)
    parser.add_argument("--a_bins", type=int, default=20)
    
    parser.add_argument('--gfn_lr', type=float, default=5e-4, help='Learning rate for GFlowNet')
    parser.add_argument('--gfn_batch_size', type=int, default=128, help='Batch size for GFlowNet')
    parser.add_argument('--gfn_num_states', type=int, default=128, help='Number of states sampled from replay buffer for GFlowNet')
    
    parser.add_argument("--use-ln-q", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    help="Enable/disable layer norm in the Q function")
    parser.add_argument("--use-ln-policy", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    help="Enable/disable layer norm in the MLP policy")
    
    args = parser.parse_args()
    # fmt: on
    return args

def make_env(env_id, seed, idx, capture_video, run_name,step_list=[250000,500000]):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}",step_trigger= lambda x: x in step_list)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(torch.cuda.is_available())

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    action_min = float(envs.single_action_space.low[0])
    action_max = float(envs.single_action_space.high[0])
    s_dim = envs.single_observation_space.shape[0]
    a_dim = envs.single_action_space.shape[0]
    
    if args.autotune:
        target_entropy = args.target_entropy#-torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
    
    gflownet = GFN(s_dim, a_dim, a_bins=args.a_bins, alpha=alpha, action_min=action_min, action_max=action_max, gfn_batch_size=args.gfn_batch_size, gfn_lr=args.gfn_lr, use_ln_q=args.use_ln_q, use_ln_policy=args.use_ln_policy).to(device)
    q_optimizer = optim.Adam(list(gflownet.q1.parameters()) + list(gflownet.q2.parameters()), lr=args.q_lr)
    
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            actions = torch.Tensor(actions).to(device)
            actions = gflownet.dequantize_action(gflownet.quantize_action(actions)).detach().cpu().numpy()
        else:
            actions, log_prob = gflownet(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()
            
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break
        
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs
        
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_actions, logp = gflownet(data.next_observations)
                q1_target = gflownet.q1_target(data.next_observations, next_actions)
                q2_target = gflownet.q2_target(data.next_observations, next_actions)
                q_target = torch.min(q1_target, q2_target)
                target = data.rewards.flatten() + args.gamma * (1 - data.dones.flatten()) * (q_target.flatten() - alpha * logp.flatten())
            q1 = gflownet.q1(data.observations, data.actions).squeeze(1)
            q2 = gflownet.q2(data.observations, data.actions).squeeze(1)
            q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()
            
            gfn_loss, logZ = gflownet.train_GFN(data.observations[:args.gfn_num_states])
            
            if args.autotune:
                alpha_loss = (-log_alpha.exp() * (logp + target_entropy)).mean()
                a_optimizer.zero_grad()
                alpha_loss.backward()
                a_optimizer.step()
                alpha = log_alpha.exp().item()
                gflownet.alpha = alpha
                
            for param, target_param in zip(gflownet.q1.parameters(), gflownet.q1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(gflownet.q2.parameters(), gflownet.q2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                
            if global_step % 100 == 0:
                writer.add_scalar("losses/q_loss", q_loss.item(), global_step)
                writer.add_scalar("losses/gfn_loss", gfn_loss, global_step)
                writer.add_scalar("losses/logZ", logZ, global_step)
                writer.add_scalar("losses/td_target", target.mean().item(), global_step)
                writer.add_scalar("charts/alpha", alpha, global_step)
                writer.add_scalar("losses/entropy", -logp.mean().item(), global_step)
                if args.autotune:
                    writer.add_scalar("charts/alpha_loss", alpha_loss.item(), global_step)