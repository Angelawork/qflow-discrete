import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob


def parse_log_files(path):
    global_steps = []
    episodic_returns = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("global_step="):
                parts = line.split(",")

                # Extract and remove non-numeric characters
                global_step = int(parts[0].split("=")[1])          
                parsed_str = parts[1].split("=")[1].strip().replace("[", "").replace("]", "")

                global_steps.append(global_step)
                episodic_returns.append(float(parsed_str))
    return np.array(global_steps), np.array(episodic_returns)

def fill_missing_values(array):
    filled = np.copy(array)
    missing_i = np.where(np.isnan(array))[0]
    for index in missing_i:
        if index == 0:
            filled[index] = filled[index+1]
        elif index == len(array) - 1:
            filled[index] = filled[index-1]
        else:
            filled[index] = (filled[index-1] + filled[index+1]) / 2
    return filled

def rolling_average(data, window_size=15):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

def Compare_plot(expr):
    sac_files = glob.glob(os.path.join(log_directory+expr+"/", "sac*.log"))
    gfn_files = glob.glob(os.path.join(log_directory+expr+"/", "gfn*.log"))

    sac_returns_all, gfn_returns_all = [], []
    for path in sac_files:
        sac_s, sac_r = parse_log_files(path)
        sac_returns_all.append(sac_r)

    for path in gfn_files:
        gfn_s, gfn_r = parse_log_files(path)
        gfn_returns_all.append(gfn_r)
    
    max_length = max(len(sac_r) for sac_r in sac_returns_all + gfn_returns_all)
    sac_padded = np.array([np.pad(sac_ret, (0, max_length - len(sac_ret)), mode="edge") for sac_ret in sac_returns_all])
    gfn_padded = np.array([np.pad(gfn_ret, (0, max_length - len(gfn_ret)), mode="edge") for gfn_ret in gfn_returns_all])

    sac_rolled = np.array([rolling_average(sac_ret) for sac_ret in sac_padded])
    gfn_rolled = np.array([rolling_average(gfn_ret) for gfn_ret in gfn_padded])
    
    # mean returns
    sac_returns = np.mean(sac_rolled, axis=0)
    gfn_returns = np.mean(gfn_rolled, axis=0)

    # mean final rewards
    sac_final_reward = np.mean([sac_ret[-1:] for sac_ret in sac_rolled])
    gfn_final_reward = np.mean([gfn_ret[-1:] for gfn_ret in gfn_rolled])

    sac_std = np.std(sac_rolled, axis=0)
    gfn_std = np.std(gfn_rolled, axis=0)

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=np.arange(len(sac_returns)), y=sac_returns, label=f"SAC (Avg Final Reward: {sac_final_reward:.2f})")
    sns.lineplot(x=np.arange(len(gfn_returns)), y=gfn_returns, label=f"GFN (Avg Final Reward: {gfn_final_reward:.2f})")
    plt.fill_between(np.arange(len(sac_returns)), sac_returns - sac_std, sac_returns + sac_std, alpha=0.4)
    plt.fill_between(np.arange(len(gfn_returns)), gfn_returns - gfn_std, gfn_returns + gfn_std, alpha=0.4)

    plt.xlabel("Recorded Steps(total = 1.5M global steps)")
    plt.ylabel("Average Episodic Returns")
    plt.title("Comparison of GFN and SAC: Padded Average Rewards in " + expr)
    plt.legend()
    plt.ticklabel_format(style="plain", axis="x")
    
    plt.show()

if __name__ == "__main__":
    log_directory = "Tuning_comparison/"
    expr =  ["Hopper","Ant","Walker2d","HalfCheetah","HumanoidStandup","Humanoid"]
    for e in expr:
        Compare_plot(e)