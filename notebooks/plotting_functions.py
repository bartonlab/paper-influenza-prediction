import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

## General
def cm2inch(x): return float(x)/2.54

plotwidth    = cm2inch(8.90)


#=======Functions for Fig-EMD_simulated_data


# Helper function to load EMD data
def load_emd_data(prefix, T0, Tf,data_dir):
    filepath = os.path.join(data_dir, f"{prefix}_T0_{T0}_tf_{Tf}.txt")
    return np.loadtxt(filepath)


#======Functions for Fig-EMD_vs_horizon_all_proteins_2012_Py======

# Trim and concat function

def trim_and_concat(vec, chunk_size, n):
    vec = np.asarray(vec)
    chunks = [vec[i:i+chunk_size] for i in range(0, len(vec), chunk_size)]
    trimmed_chunks = [chunk[n-1:] for chunk in chunks if len(chunk) >= n]  # n-1 because Python is 0-based
    return np.concatenate(trimmed_chunks)



# Group labels: 6, 9, 12 repeated for each chunk
def create_dataframe(data, label):
    if len(data) % 3 != 0:
        raise ValueError(f"Length of data for {label} is not divisible by 3")
    n_repeats = len(data) // 3
    horizons = [6, 9, 12] * n_repeats
    return pd.DataFrame({'Horizon': horizons, 'Value': data, 'Label': label})

def create_dataframe_irregular(data, label):
    n = len(data)
    sizes = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
    assert sum(sizes) == n, "Partition sizes must sum to total length"

    split_data = []
    start = 0
    for size in sizes:
        split_data.append(data[start:start+size])
        start += size

    horizons = [6, 9, 12]
    df = pd.DataFrame({
        'Horizon': [h for h, group in zip(horizons, split_data) for _ in group],
        'Value': np.concatenate(split_data),
        'Label': label
    })
    return df




#======Functions for Fig-AUC_vs_f_all_subtypes======

# Function to load and flatten a file (equivalent to Julia's `[:]`)
def load_flat(path):
    return np.loadtxt(path).flatten()

# Function to load data for a given subtype (e.g. "H3N2") and protein (e.g. "HA" or "NA")
def load_data(subtype, protein,base_auc,base_fix):
    auc_path = os.path.join(base_auc, subtype, protein)
    fix_path = os.path.join(base_fix, subtype, protein)

    return {
        "auc_sim": load_flat(os.path.join(auc_path, "AUC_vs_f0_sim.txt")),
        "auc_null": load_flat(os.path.join(auc_path, "AUC_vs_f0_null.txt")),
        "fix_top": load_flat(os.path.join(fix_path, "fix_sim_score_top.txt")),
        "fix_top25": load_flat(os.path.join(fix_path, "fix_sim_score_top_25.txt")),
        "fix_bottom": load_flat(os.path.join(fix_path, "fix_sim_score_bottom.txt")),
        "fix_bottom25": load_flat(os.path.join(fix_path, "fix_sim_score_bottom_25.txt")),
        "fix_null": load_flat(os.path.join(fix_path, "fix_prob_frequency_null.txt")),
    }


def plot_auc_bar(data, label, ax=None):

    frequencies = np.arange(0.1, 1.0, 0.1)

    bar_width = 0.09

    # Colors
    blue_main = (0.3, 0.45, 0.8)
    gray = 'lightgray'
    # Create new axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5, 4.5), dpi=100)
    else:
        fig = ax.figure

    # Plot bars
    ax.bar(frequencies, data["auc_sim"], width=bar_width, color=blue_main, label="prediction")
    ax.bar(frequencies, data["auc_null"], width=bar_width, color=gray, label="null model")

    ax.set_ylim(0.0, 1.01)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("frequency")
    ax.set_ylabel("Fix/loss AUROC")
    ax.set_title(label, loc='left')
    ax.legend(loc='upper right', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, ax

#======Functions for Fig-Fix_prob_vs_f_all_subtypes======


def plot_fixation_probability(data, label, ax=None):
    frequencies = np.arange(0.1, 1.0, 0.1)

    # Style
    lw = 2

    # Colors
    blue_main = (0.3, 0.45, 0.8)
    blue_light = (0.3, 0.7, 1.0)
    red_light = (0.95, 0.4, 0.4)
    red_dark = (0.6, 0.05, 0.05)
    gray = 'lightgray'

    # Create new axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5, 4.5), dpi=100)
    else:
        fig = ax.figure

    # Plot lines
    ax.plot(frequencies, frequencies, linestyle='--', color='black', label="")
    ax.plot(frequencies, data["fix_top25"],   color=blue_light, linewidth=lw, label="top 25%")
    ax.plot(frequencies, data["fix_top"],     color=blue_main,  linewidth=lw, label="top 50%")
    ax.plot(frequencies, data["fix_bottom25"],color=red_light,  linewidth=lw, label="bottom 25%")
    ax.plot(frequencies, data["fix_bottom"],  color=red_dark,   linewidth=lw, label="bottom 50%")
    ax.plot(frequencies, data["fix_null"],    color=gray,       linewidth=lw, label="all mutations")

    ax.set_xlim(0.0, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("frequency")
    ax.set_ylabel("Fixation probability")
    ax.set_title(label, loc='left')
    ax.legend(loc=(0.01,0.75), frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, ax

#======Functions for Fig-EMD_vs_t_all_subtypes======
# Colors
blue = (0.3, 0.45, 0.8)
green = 'green'
gray = 'gray'
def plot_panel(ax, x, y_null, y_lb, y_pred, title, ylabel="", xlabel="", xticks=None, yticks=None, yticklabels=True, xticklabels=None):
    ax.plot(x, y_null, color=gray, linewidth=2, marker='o', markersize=3, label="null")
    ax.plot(x, y_lb, color=green, linewidth=2, marker='^', markersize=3, alpha=0.5, label="lower bound")
    ax.plot(x, y_pred, color=blue, linewidth=2, marker='D', markersize=3, label="prediction")
    ax.set_title(title, loc='left')
    ax.set_ylim(0, 12)

    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if yticks is not None:
        ax.set_yticks(yticks)
        if not yticklabels:
            ax.set_yticklabels([])
    if xticks is not None:
        ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels, rotation=35)
        else:
            ax.set_xticklabels([""] * len(xticks))

    # Keep x and y axes, remove top and right frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(left=True, bottom=True)
    ax.grid(False)
    


#======Functions for Fig-Enrichment======

def plot_enrichment_series(ax, enrichment, ylims, yticks, label_colors, panel_title,
                           show_xlabel=True, show_ylabel=True, xlims=(0.8, 20.2)):
    for label, enr in enrichment.items():
        ax.plot(range(1, len(enr)+1), enr, label=label, linewidth=2, alpha=0.8,
                color=label_colors[label])
        ax.fill_between(range(1, len(enr)+1), enr, alpha=0.3, color=label_colors[label])

    ax.set_ylim(*ylims)
    ax.set_xlim(*xlims)
    ax.set_yticks(yticks)
    ax.set_xticks([1, 10, 20])
    if show_ylabel:
        ax.set_ylabel("Fold enrichment\nfor epitope positions")
    if show_xlabel:
        ax.set_xlabel("Fraction of most positively selected variants (%)")
    ax.set_title(panel_title, loc='left')
    ax.legend(loc="upper right", frameon=False)
    # Keep x and y axes, remove top and right frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
def get_enrichment_plot_H3N2_HA(enrichment_dict_,ax, show_xlabel=True, show_ylabel=True):
    colors = {
        "RBS - 35": "cyan",
        "Koel et.al - 7": "blue",
        "Lukza et.al - 129": "green",
        "Wolf et.al - 49": "orange"
    }
    plot_enrichment_series(
        ax,
        enrichment_dict_,
        ylims=(0, 36),
        yticks=[1, 16, 32],
        label_colors=colors,
        panel_title="a. H3N2-HA",
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel
    )

def get_enrichment_plot_H1N1_HA(enrichment_dict_,ax, show_xlabel=True, show_ylabel=True):
    colors = {
        "RBS - 20": "cyan",
        "Globular-head-Canton et.al - 50": "blue"
    }
    plot_enrichment_series(
        ax,
        enrichment_dict_,
        ylims=(0, 6),  # updated range
        yticks=[1, 3, 5],
        label_colors=colors,
        panel_title="b. H1N1-HA",
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel
    )

def get_enrichment_plot_H3N2_NA(enrichment_dict_,ax, show_xlabel=True, show_ylabel=True):
    colors = {
        "Munoz et.al - 161": "blue",
        "Bhatt et.al - 223": "green"
    }
    plot_enrichment_series(
        ax,
        enrichment_dict_,
        ylims=(0, 5),
        yticks=[1, 3, 5],
        label_colors=colors,
        panel_title="c. H3N2-NA",
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel    )

def get_enrichment_plot_H1N1_NA(enrichment_dict_,ax, show_xlabel=True, show_ylabel=True):
    colors = {
        "Munoz et.al - 122": "blue",
        "Bhatt et.al - 113": "green"
    }
    plot_enrichment_series(
        ax,
        enrichment_dict_,
        ylims=(0, 5),
        yticks=[1, 3, 5],
        label_colors=colors,
        panel_title="d. H1N1-NA",
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel
    )

# === Final layout ===
def make_all_enrichment_layout():
    fig, axs = plt.subplots(2, 2, figsize=(2*1.5*plotwidth, 1.5*plotwidth), constrained_layout=True)
    

    get_enrichment_plot_H3N2_HA(axs[0, 0], show_xlabel=False, show_ylabel=True)
    get_enrichment_plot_H1N1_HA(axs[0, 1], show_xlabel=False, show_ylabel=False)

    get_enrichment_plot_H3N2_NA(axs[1, 0], show_xlabel=True, show_ylabel=True)
    get_enrichment_plot_H1N1_NA(axs[1, 1], show_xlabel=True, show_ylabel=False)

    return fig, axs    
    

