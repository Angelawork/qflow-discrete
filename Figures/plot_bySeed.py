import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def parse_logs(files):
    global_steps = []
    returns = []
    for path in files:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("global_step="):
                    parts = line.split(",")
                    global_step = int(parts[0].split("=")[1])
                    parsed_str = parts[1].split("=")[1].strip().replace("[", "").replace("]", "")
                    global_steps.append(global_step)
                    returns.append(float(parsed_str))
    return np.array(global_steps), np.array(returns)

def rolling_avg(data, window=5):
    return np.convolve(data, np.ones(window) / window, mode='valid')

def plot_comparison(sac_file, gfn_file, ax, expr, seed):
    sns.set(style="whitegrid")
    
    sac_steps, sac_returns = parse_logs([sac_file])
    gfn_steps, gfn_returns = parse_logs([gfn_file])

    roll_w = 15
    sac_avg = rolling_avg(sac_returns,window=roll_w)
    gfn_avg = rolling_avg(gfn_returns,window=roll_w)
    
    ax.plot(sac_steps[:len(sac_avg)], sac_avg, label=f"SAC Avg Final reward: {sac_avg[-1]:.2f}")
    ax.plot(gfn_steps[:len(gfn_avg)], gfn_avg, label=f"GFN Avg Final reward: {gfn_avg[-1]:.2f}")

    ax.set_xlabel("Global Steps")
    ax.set_ylabel("Average Episodic Returns")
    ax.set_title(f"Comparison: Rolling Avg(window={roll_w}) Rewards in {expr} {seed}")
    ax.legend()
    ax.ticklabel_format(style="plain", axis="x")
    return sac_avg[-1], gfn_avg[-1]

def compare_plot(expr):
    sac_files = glob.glob(os.path.join(log_directory + expr, "sac*.log"))
    gfn_files = glob.glob(os.path.join(log_directory + expr, "gfn*.log"))
    sac_final_avg = []
    gfn_final_avg = []
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(3, 1, figsize=(14, 18))

    for i, (sac_seed, gfn_seed) in enumerate(zip(["seed42", "seed128", "seed456"], ["seed42", "seed128", "seed456"])):
        sac_file = next(file for file in sac_files if sac_seed in file)
        gfn_file = next(file for file in gfn_files if gfn_seed in file)
        sac, gfn = plot_comparison(sac_file, gfn_file, axs[i], expr, sac_seed)
        sac_final_avg.append(sac)
        gfn_final_avg.append(gfn)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35)
    fig.legend([f"Mean SAC Avg Final reward: {np.mean(sac_final_avg):.2f}", 
                f"Mean GFN Avg Final reward: {np.mean(gfn_final_avg):.2f}"], loc="upper left")
    plt.show()

if __name__ == "__main__":
    log_directory = "Tuning_comparison/"
    expr = ["Hopper","Ant","Walker2d","HalfCheetah","HumanoidStandup","Humanoid"]
    for e in expr:
        compare_plot(e)
