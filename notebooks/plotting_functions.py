import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


# === Settings ===
base = 9
matplotlib.rcParams.update({
    'font.size': base,
    'axes.labelsize': base,
    'axes.titlesize': base,
    'xtick.labelsize': base,
    'ytick.labelsize': base,
    'legend.fontsize': base,
    'font.family': 'Arial',
})
ms = 6
lw=2
## General

def cm2inch(x): return float(x)/2.54

plotwidth    = cm2inch(8.90)



def get_figsize(layout="horizontal", width_cm=8.9):
    if layout == "horizontal":
        return (1.5*plotwidth, 0.5*plotwidth)
    elif layout == "vertical":
        return (1.5*plotwidth, 1.0*plotwidth)
    elif layout == "square":
        return (1.5*plotwidth, 1.5*plotwidth)
    elif layout == "golden":
        return (cm2inch(width_cm), cm2inch(width_cm * 0.618))
    else:
        raise ValueError("Unknown layout")

#=======Functions for allele frequency plots

# Helper function to select trajectory
def select_traj(freq_vs_time, pos, state, q):
    if state > q:
        raise ValueError("State must be in [1, q]")
    return freq_vs_time[(pos - 1) * q + state - 1, :]

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
    ax.set_xticks([0.0,0.5,1.0])

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticks([0.0, 0.5, 1.0]) 

    ax.set_title(label, loc='left')
    ax.legend(loc=(0.05,0.77), handlelength=1.0, handleheight=0.5, handletextpad=0.2, labelspacing=0.3 ,frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, ax

#======Functions for Fig-Fix_prob_vs_f_all_subtypes======


def plot_fixation_probability(data, label, ax=None):
    frequencies = np.arange(0.1, 1.0, 0.1)

    # Style
    lw = 1

    # Colors
    blue_main = (0.3, 0.45, 0.8)
    blue_light = (0.3, 0.7, 1.0)
    red_light = (0.95, 0.4, 0.4)
    red_dark = (0.6, 0.05, 0.05)
    gray = 'lightgray'

    # Create new axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=100)
    else:
        fig = ax.figure

    # Plot lines
    ax.plot(frequencies, frequencies, linestyle='--', color='black', linewidth=lw,label="")
    ax.plot(frequencies, data["fix_top25"],   color=blue_light, linewidth=lw, label="top 25%")
    ax.plot(frequencies, data["fix_top"],     color=blue_main,  linewidth=lw, label="top 50%")
    ax.plot(frequencies, data["fix_bottom25"],color=red_light,  linewidth=lw, label="bottom 25%")
    ax.plot(frequencies, data["fix_bottom"],  color=red_dark,   linewidth=lw, label="bottom 50%")
    ax.plot(frequencies, data["fix_null"],    color=gray,       linewidth=lw, label="all mutations")

    ax.set_xlim(0.0, 1.01)
    ax.set_xticks([0.0, 0.5, 1.0]) 
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticks([0.0, 0.5, 1.0]) 
    ax.set_title(label, loc='left')
    #ax.legend(loc=(0.01,0.75), frameon=False)
    ax.legend(loc=(0.7, 0.0),handlelength=0.5, handleheight=0.5, handletextpad=0.2, labelspacing=0.1,  frameon=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, ax

#======Functions for Fig-EMD_vs_t_all_subtypes======
# Colors
blue = (0.3, 0.45, 0.8)
green = 'green'
gray = 'gray'
def plot_panel(ax, x, y_null, y_lb, y_pred, title, ylabel="", xlabel="", xticks=None, yticks=None, yticklabels=True, xticklabels=None):
    ax.plot(x, y_null, color=gray, linewidth=1, marker='o', markersize=1, alpha=0.8,label="null")
    ax.plot(x, y_lb, color=green, linewidth=1, marker='^', markersize=1, alpha=0.8, label="lower bound")
    ax.plot(x, y_pred, color=blue, linewidth=1, marker='D', markersize=1,alpha=0.8, label="prediction")
    ax.set_title(title, loc='left')
    ax.set_ylim(0, 12)
# Inside plot_panel, or directly for the ax:
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

    
#======Functions for Clades evolution Figure======

def get_contrast_color(rgb_color):
    # Convert to luminance using standard formula
    r, g, b = rgb_color[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'black' if luminance > 0.6 else 'white'


def get_clade_color_map(all_clades, colormap='tab20'):
    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, len(all_clades)))
    return dict(zip(sorted(all_clades), colors))


def plot_clade_dynamics_with_fitness(df_pop, clade_color_map, title='', normalize_to_100=True, label_threshold=0.001,ax=None):
    df = df_pop.copy()
    df['clade'] = df['clade'].astype(str).str.strip()
    df['Date'] = pd.to_datetime(df['Date'])

    if normalize_to_100:
        df['frequency'] = df.groupby('Date')['frequency'].transform(lambda x: 100 * x / x.sum())
    else:
        df['frequency'] = df.groupby('Date')['frequency'].transform(lambda x: x / x.sum())

    pivot = df.pivot_table(index='Date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot = pivot.sort_index()

    max_freqs = pivot.max()
    clades_to_plot = max_freqs[max_freqs > (label_threshold * 100 if normalize_to_100 else label_threshold)].index
    pivot = pivot[clades_to_plot]

    x = pivot.index
    Y = pivot.values.T
    clades = pivot.columns

    fitness_pivot = df.pivot_table(index='Date', columns='clade', values='fitness', aggfunc='mean').fillna(0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.5))
    else:
        fig = ax.figure

    colors = [clade_color_map[clade] for clade in clades]

    ax.stackplot(x, Y, labels=clades, colors=colors)
    ax.set_title(title, loc='left')
    ax.set_ylabel('Frequency (%)' if normalize_to_100 else 'Frequency')
    ax.set_xlabel('Observation date (days)')
    ax.set_ylim(0, 100 if normalize_to_100 else 1)

    # --- Match x-tick logic from four-forecast plot ---
    tick_positions = pd.date_range(start=x[0], end=x[-1], freq='3M')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in tick_positions], rotation=35)

    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.02), title="Clade", ncol=1,handletextpad=0.2, columnspacing=0.3, handlelength=1.0,frameon=False  )

    cumulative_Y = np.cumsum(Y, axis=0)
    base_Y = np.vstack([np.zeros(cumulative_Y.shape[1]), cumulative_Y[:-1]])

    def get_contrast_color(rgb_color):
        r, g, b = rgb_color[:3]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return 'black' if luminance > 0.6 else 'white'

    ax.set_xlim(x[0], x[-1])
    return fig, ax

    
def plot_clade_dynamics_with_four_forecasts(df_past, df_forecast_rm, df_forecast1, df_forecast2, df_forecast3,
                                            title='', clade_color_map=None, colormap='tab20',
                                            normalize_to_100=True, label_threshold=0.001,ax=None):
    df_past = df_past.copy()
    df_forecast_rm = df_forecast_rm.copy()
    df_forecast1 = df_forecast1.copy()
    df_forecast2 = df_forecast2.copy()
    df_forecast3 = df_forecast3.copy()

    for df in [df_past, df_forecast_rm, df_forecast1, df_forecast2, df_forecast3]:
        df['Date'] = pd.to_datetime(df['Date'])

    df = pd.concat([df_past, df_forecast_rm, df_forecast1, df_forecast2, df_forecast3])
    df['clade'] = df['clade'].astype(str).str.strip()

    if normalize_to_100:
        df['frequency'] = df.groupby('Date')['frequency'].transform(lambda x: 100 * x / x.sum())
    else:
        df['frequency'] = df.groupby('Date')['frequency'].transform(lambda x: x / x.sum())

    pivot = df.pivot_table(index='Date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot = pivot.sort_index()

    max_freqs = pivot.max()
    clades_to_plot = max_freqs[max_freqs > (label_threshold * 100 if normalize_to_100 else label_threshold)].index
    pivot = pivot[clades_to_plot]

    x = pivot.index
    Y = pivot.values.T
    clades = pivot.columns

    fitness_pivot = df.pivot_table(index='Date', columns='clade', values='fitness', aggfunc='mean').fillna(0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.5))
    else:
        fig = ax.figure

    # === Use clade_color_map if provided ===
    if clade_color_map is not None:
        colors = [clade_color_map.get(clade, '#cccccc') for clade in clades]  # default to gray if missing
    else:
        cmap = plt.get_cmap(colormap)
        n_colors = len(clades)
        colors = cmap(np.linspace(0, 1, n_colors))

    ax.stackplot(x, Y, labels=clades, colors=colors)
    ax.set_title(title,loc='left')
    ax.set_ylabel('Frequency (%)' if normalize_to_100 else 'Frequency')
    ax.set_xlabel('Observation date (days)')
    ax.set_ylim(0, 100 if normalize_to_100 else 1)

    forecasts = [
        (df_forecast_rm['Date'].min(), df_forecast_rm['Date'].max(), '3m'),
        (df_forecast1['Date'].min(), df_forecast1['Date'].max(), '6m'),
        (df_forecast2['Date'].min(), df_forecast2['Date'].max(), '9m'),
        (df_forecast3['Date'].min(), df_forecast3['Date'].max(), '12m'),
    ]

    past_end = df_past['Date'].max()
    ax.axvline(past_end, color='black', linestyle='--', linewidth=1.5, zorder=3,alpha=1.0)

    tick_positions = set(pd.date_range(start=x[0], end=past_end, freq='3M'))
    forecast_start_dates = []

    for f_start, f_end, label in forecasts:
        ax.axvline(f_end, color='black', linestyle='--', linewidth=1.0, zorder=3,alpha=0.5)
        ax.axvspan(f_start, f_end, color='gray', alpha=0.2, zorder=0)
        ax.text(f_end, ax.get_ylim()[1] + 2, label, ha='center', va='bottom')
        forecast_start_dates.append(f_start)

    tick_positions.update(forecast_start_dates)
    tick_positions = sorted(tick_positions)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([pd.to_datetime(d).strftime('%Y-%m-%d') for d in tick_positions], rotation=35)

  
    cumulative_Y = np.cumsum(Y, axis=0)
    base_Y = np.vstack([np.zeros(cumulative_Y.shape[1]), cumulative_Y[:-1]])

    def get_contrast_color(rgb_color):
        if isinstance(rgb_color, str):
            rgb_color = plt.colors.to_rgb(rgb_color)
        r, g, b = rgb_color[:3]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return 'black' if luminance > 0.6 else 'white'

    for idx, clade in enumerate(clades):
        y_vals = Y[idx]
        clade_color = colors[idx]
        text_color = get_contrast_color(clade_color)
        fit_font = {'color': text_color}

        if np.max(y_vals) >= (label_threshold * 100 if normalize_to_100 else label_threshold):
            max_idx = np.argmax(y_vals)
            shift = 5 if max_idx > len(x) - 6 else 0
            label_idx = max(0, max_idx - shift)

            date = x[label_idx] 
            y_base = base_Y[idx, label_idx]
            y_top = cumulative_Y[idx, label_idx]
            y_middle = y_base + (y_top - y_base) / 2

            fitness_val = fitness_pivot.loc[x[label_idx], clade]
            fitness_label = f"{fitness_val:.2f}" if fitness_val != 0 else "h: NA"

            if clade == '3C':
                date_shifted = date + pd.Timedelta(days=50)
                y_middle -= 3
            else:
                date_shifted = date - pd.Timedelta(days=20)

            ax.text(date_shifted - pd.Timedelta(days=7), y_middle - 2.5, fitness_label, ha='center', va='bottom', fontsize=7,**fit_font)

    #plt.tight_layout()
    ax.set_xlim(x[0], x[-1])
    return fig, ax


