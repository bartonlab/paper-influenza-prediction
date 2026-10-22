"""
This Python file contains code for generating plots in:
    Fitness inference from complex evolutionary histories with genetic linkage
    Muhammad S. Sohail
    Raymond H. Y. Louie
    Matthew R. McKay (m.mckay@ust.hk)
    John P. Barton (john.barton@ucr.edu)

Additional code to pre-process the data and pass it to these plotting
routines is contained in the Jupyter notebook `figures.ipynb`.
"""

#############  PACKAGES  #############

import sys, os
from copy import deepcopy

import numpy as np

import scipy as sp
import scipy.stats as st

import pandas as pd

import datetime as dt

from sklearn.metrics import roc_auc_score

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

from PIL import Image 

import seaborn as sns

from colorsys import hls_to_rgb

from . import mplot as mp


############# PARAMETERS #############

# GLOBAL VARIABLES

NUC = ['-', 'A', 'C', 'G', 'T']
REF = NUC[0]
EXT = '.pdf'

# GitHub directories
DATA_DIR = 'data'
FIG_DIR  = 'figures'

# Standard color scheme

BKCOLOR  = '#252525'
LCOLOR   = '#969696'
C_BEN    = '#EB4025' #'#F16913'
C_BEN_LT = '#F08F78' #'#fdd0a2'
C_NEU    =  LCOLOR   #'#E8E8E8' # LCOLOR
C_NEU_LT = '#E8E8E8' #'#F0F0F0' #'#d9d9d9'
C_DEL    = '#3E8DCF' #'#604A7B'
C_DEL_LT = '#78B4E7' #'#dadaeb'
C_MPL    = '#FFB511'
C_SL     = C_NEU

C_SLIDE   = '#F08F78'
C_SLIDE_1 = '#78B4E7'

# Plot conventions

def cm2inch(x): return float(x)/2.54
SINGLE_COLUMN   = cm2inch(8.8)
ONE_FIVE_COLUMN = cm2inch(12.1)
DOUBLE_COLUMN   = cm2inch(18.4)
# SINGLE_COLUMN   = cm2inch(8.8)
# ONE_FIVE_COLUMN = cm2inch(11.4)
# DOUBLE_COLUMN   = cm2inch(18.0)
SLIDE_WIDTH     = 10.5

GOLDR      = (1.0 + np.sqrt(5)) / 2.0
TICKLENGTH = 3
TICKPAD    = 3
AXWIDTH    = 0.4

# # paper style
# FONTFAMILY   = 'Helvetica' #'Arial'
# SIZESUBLABEL = 10 #8
# SIZELABEL    = 8 #6
# SIZETICK     = 8 #6
# SMALLSIZEDOT = 6.
# SIZELINE     = 0.6

# # grant style
# FONTFAMILY   = 'Arial'
# SIZESUBLABEL = 12
# SIZELABEL    = 8
# SIZETICK     = 8
# SMALLSIZEDOT = 6. * 1.3
# SIZELINE     = 0.6

# # slides style
# FONTFAMILY   = 'Avenir'
# SIZESUBLABEL = 18
# SIZELABEL    = 18
# SIZETICK     = 18
# SMALLSIZEDOT = 6. * 7
# SIZELINE     = 1.5

# poster style
FONTFAMILY   = 'Avenir'
SIZESUBLABEL = 24
SIZELABEL    = 24
SIZETICK     = 24
SMALLSIZEDOT = 6. * 7
SIZELINE     = 1.5


FIGPROPS = {
    'transparent' : True,
    #'bbox_inches' : 'tight'
}

DEF_BARPROPS = {
    'lw'          : SIZELINE/2,
    'width'       : 0.25,
    'edgecolor'   : BKCOLOR,
    'align'       : 'center', #other option: edge
    'orientation' : 'vertical'
}

DEF_HISTPROPS = {
    'histtype'    : 'bar',
    'lw'          : SIZELINE/2,
    'rwidth'      : 0.8,
    'ls'          : 'solid',
    'edgecolor'   : BKCOLOR,
    'alpha'       : 0.5
}

DEF_ERRORPROPS = {
    'mew'        : AXWIDTH,
    'markersize' : SMALLSIZEDOT/2,
    'fmt'        : 'o',
    'elinewidth' : SIZELINE/2,
    'capthick'   : 0,
    'capsize'    : 0
}

DEF_LINEPROPS = {
    'lw' : SIZELINE,
    'ls' : '-'
}

DEF_LABELPROPS = {
    'family' : FONTFAMILY,
    'size'   : SIZELABEL,
    'color'  : BKCOLOR
}

DEF_SUBLABELPROPS = {
    'family'     : FONTFAMILY,
    'size'       : SIZESUBLABEL,
    'fontweight' : 'bold',
    'ha'         : 'center',
    'va'         : 'center',
    'color'      : 'k',
    'clip_on'    : False
}

DEF_TICKLABELPROPS = {
    'family' : FONTFAMILY,
    'size'   : SIZETICK,
    'color'  : BKCOLOR
}

DEF_TICKPROPS = {
    'length'    : TICKLENGTH,
    'width'     : AXWIDTH/2,
    'pad'       : TICKPAD,
    'axis'      : 'both',
    'direction' : 'out',
    'colors'    : BKCOLOR,
    'bottom'    : True,
    'left'      : True,
    'top'       : False,
    'right'     : False
}

DEF_MINORTICKPROPS = {
    'length'    : TICKLENGTH-1.25,
    'width'     : AXWIDTH/2,
    'axis'      : 'both',
    'direction' : 'out',
    'which'     : 'minor',
    'color'     : BKCOLOR
}

DEF_AXPROPS = {
    'linewidth' : AXWIDTH,
    'linestyle' : '-',
    'color'     : BKCOLOR
}

PARAMS = {'text.usetex': False, 'mathtext.fontset': 'stixsans', 'mathtext.default' : 'regular'}
plt.rcParams.update(PARAMS)


#___ MAIN PLOTS


def generate_overview_data(n_vars=10, N=1000, t_pred=200, t_final=500, gamma=1, n_sims=5, seed=0):
    """ Perform Wright-Fisher simulations to generate data for the overview figure. """

    # Wright-Fisher simulation
    rng = np.random.default_rng(seed)
    def wf(x, s, N):
        p = x * (1 + s) / np.sum(x * (1 + s))
        n = rng.multinomial(N, p)
        return n / N
    
    # Initial simulation
    N = 1000        # Population size
    n_vars = 10     # Number of variants
    s = rng.uniform(-0.025, 0.025, size=n_vars)  # Selection coefficient
    x_init = rng.uniform(0.01, 0.99, size=n_vars)  # Initial frequencies
    x_init /= np.sum(x_init)  # Normalize initial frequencies
    x_vec = [x_init]
    t_vec = [0]
    for t in range(t_pred-1):
        x = wf(x_vec[-1], s, N)
        x_vec.append(x)
        t_vec.append(t+1)

    x_vec = np.array(x_vec)
    t_vec = np.array(t_vec)

    # Save data for inference
    c_int = np.zeros((n_vars, n_vars))

    # Save data to CSV file
    freq_file = '%s/overview-frequencies-true.csv' % FIG_DIR
    f = open(freq_file, 'w')
    f.write('variant,time,frequency\n')
    for i in range(len(x_vec)):
        for j in range(len(x_vec[i])):
            f.write('%d,%d,%.6f\n' % (j, t_vec[i], x_vec[i][j]))

            c_int[j, j] += x_vec[i][j] * (1 - x_vec[i][j])  # Variance of frequencies
            for k in range(j+1, len(x_vec[i])):
                c_int[j, k] -= x_vec[i][j] * x_vec[i][k]
                c_int[k, j] = c_int[j, k]
    f.close()

    # Infer selection coefficient and its uncertainty
    dx = x_vec[-1]-x_vec[0]
    c_int = np.sum(x_vec * (1 - x_vec))
    s_inf = dx / (c_int + gamma)
    s_inf_err = np.sqrt(1/(c_int + gamma)/(0.1*N))

    # Draw selection coefficients from posterior distribution and simulate future trajectories
    freq_file = '%s/overview-frequencies-sim.csv' % FIG_DIR
    f = open(freq_file, 'w')
    f.write('sim,variant,time,frequency\n')
    for i in range(n_sims):
        s = rng.normal(s_inf, s_inf_err)
        x = x_vec[-1]
        for t in range(t_final-t_pred):
            x = wf(x, s, N)

            for j in range(len(x)):
                f.write('%d,%d,%d,%.6f\n' % (i, j, t_pred+t, x[j]))
    f.close()


def plot_overview_schematic_true(freq_file, fig_name='fig-overview-true.pdf'):
    """ Plot simulated trajectories """

    # Read in data
    df = pd.read_csv(freq_file, comment='#', memory_map=True)
    vars = df['variant'].unique()
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = 1.6
    goldh = 0.7
    fig   = plt.figure(figsize=(w, goldh))

    box_top  = 0.95
    box_bot  = 0.25
    box_traj = dict(left=0.18, right=0.93, top=box_top, bottom=box_bot)
    gs_traj  = gridspec.GridSpec(1, 1, **box_traj)
    ax_traj  = plt.subplot(gs_traj[0, 0])

    # a -- trajectories

    lineprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 0.7 }
    colors = sns.husl_palette(len(vars))

    pprops = { 'xticks'      : [0, 100, 200, 300, 400, 500],
               'yticks'      : [0, 1],
               'yminorticks' : [0.25, 0.5, 0.75],
               'ylim'        : [0, 1.05],
            #    'xlabel'      : 'Time',
            #    'ylabel'      : 'Frequency',
               'plotprops'   : lineprops,
               'axoffset'    : 0.1,
               'theme'       : 'open' }

    xdat = [df['time'].unique() for v in vars]
    ydat = [df[df['variant']==v]['frequency'].values for v in vars]
    mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=colors, **pprops)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_overview_schematic_sim(freq_file, fig_name='fig-overview-sim.pdf'):
    """ Plot simulated trajectories """

    # Read in data
    df = pd.read_csv(freq_file, comment='#', memory_map=True)
    vars = df['variant'].unique()
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = 1.6
    goldh = 0.7
    fig   = plt.figure(figsize=(w, goldh))

    box_top  = 0.95
    box_bot  = 0.25
    box_traj = dict(left=0.18, right=0.93, top=box_top, bottom=box_bot)
    gs_traj  = gridspec.GridSpec(1, 1, **box_traj)
    ax_traj  = plt.subplot(gs_traj[0, 0])

    # a -- trajectories

    lineprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 0.25 }
    colors = sns.husl_palette(len(vars))

    pprops = { 'xticks'      : [0, 100, 200, 300, 400, 500],
               'yticks'      : [0, 1],
               'yminorticks' : [0.25, 0.5, 0.75],
               'ylim'        : [0, 1.05],
            #    'xlabel'      : 'Time',
            #    'ylabel'      : 'Frequency',
               'plotprops'   : lineprops,
               'axoffset'    : 0.1,
               'theme'       : 'open' }

    # Plot each simulation separately
    sims = df['sim'].unique()
    for sim in sims[:-1]:
        df_sim = df[df['sim'] == sim]
        xdat = [df_sim['time'].unique() for v in vars]
        ydat = [df_sim[df_sim['variant']==v]['frequency'].values for v in vars]
        mp.line(ax=ax_traj, x=xdat, y=ydat, colors=colors, **pprops)

    df_sim = df[df['sim'] == sims[-1]]
    xdat = [df_sim['time'].unique() for v in vars]
    ydat = [df_sim[df_sim['variant']==v]['frequency'].values for v in vars]
    mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=colors, **pprops)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def generate_overview_data_fixloss(N=1000, seed=0):
    """ Perform Wright-Fisher simulations to generate data for the overview figure. """

    # Wright-Fisher simulation
    rng = np.random.default_rng(seed)
    def wf(x, s, N):
        p = x * (1 + s) / (x * (1 + s) + (1 - x))
        n = rng.binomial(N, p)
        return n / N
    
    # Initial simulation
    N = 1000 # Population size
    s = 0
    x_init = 0.01
    x_vec = [x_init]
    t_vec = [0]
    found_enter = False
    while not found_enter:
        x = wf(x_vec[-1], s, N)
        x_vec.append(x)
        t_vec.append(t_vec[-1]+1)
        if x>=0.05:
            found_enter = True
        elif x<=0.001:
            x_vec = [x_init]
            t_vec = [0]

    found_fix = False
    x_vec_fix = [k for k in x_vec]
    t_vec_fix = [t for t in t_vec]
    s = 0.02
    while not found_fix:
        x = wf(x_vec_fix[-1], s, N)
        x_vec_fix.append(x)
        t_vec_fix.append(t_vec_fix[-1]+1)
        if x>=0.95:
            found_fix = True
        elif x<=0.01:
            x_vec_fix = [k for k in x_vec]
            t_vec_fix = [t for t in t_vec]

    found_lost = False
    x_vec_lost = [k for k in x_vec]
    t_vec_lost = [t for t in t_vec]
    s = 0
    while not found_lost:
        x = wf(x_vec_lost[-1], s, N)
        x_vec_lost.append(x)
        t_vec_lost.append(t_vec_lost[-1]+1)
        if x<=0.01:
            found_lost = True
        elif x>=0.55:
            x_vec_lost = [k for k in x_vec]
            t_vec_lost = [t for t in t_vec]

    # Save data to CSV file
    freq_file = '%s/overview-frequencies-fixloss.csv' % FIG_DIR
    f = open(freq_file, 'w')
    f.write('variant,time,frequency\n')
    for i in range(len(x_vec_fix)):
        f.write('%d,%d,%.6f\n' % (0, t_vec_fix[i], x_vec_fix[i]))
    for i in range(len(x_vec_lost)):
        f.write('%d,%d,%.6f\n' % (1, t_vec_lost[i], x_vec_lost[i]))
    f.close()


def plot_overview_schematic_fixloss(freq_file, fig_name='fig-overview-fixloss.pdf'):
    """ Plot simulated trajectories """

    # Read in data
    df = pd.read_csv(freq_file, comment='#', memory_map=True)
    vars = df['variant'].unique()
    
    # PLOT FIGURE
    
    ## set up figure grid

    w     = SLIDE_WIDTH
    goldh = 0.3 * w
    fig   = plt.figure(figsize=(w, goldh))

    box_top  = 0.95
    box_bot  = 0.10
    box_traj = dict(left=0.05, right=0.95, top=box_top, bottom=box_bot)
    gs_traj  = gridspec.GridSpec(1, 1, **box_traj)
    ax_traj  = plt.subplot(gs_traj[0, 0])

    # a -- trajectories

    lineprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 1.0 }
    colors = sns.husl_palette(len(vars))

    pprops = { 'xticks'      : [0, 100, 200, 300],
               'xminorticks' : [50, 150, 250, 350],
               'yticks'      : [0, 1],
               'yminorticks' : [0.25, 0.5, 0.75],
               'ylim'        : [0, 1.05],
            #    'xlabel'      : 'Time',
            #    'ylabel'      : 'Frequency',
               'plotprops'   : lineprops,
               'axoffset'    : 0.1,
               'theme'       : 'open' }

    xdat = [df[df['variant']==v]['time'].values for v in vars]
    ydat = [df[df['variant']==v]['frequency'].values for v in vars]
    mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=colors, **pprops)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_selection_structure_space(s_files_mut, s_files_smooth, min_counts=2, colors=None, fig_name='fig-selection-structure-space.pdf'):
    ''' Combine plots of selection coefficients shown on HA structure and enrichment in beneficial mutations at functional sites (plot_frequency_selection_structure)
        with plots of selection across HA and NA sequences (plot_selection_space_time). '''

    # PLOT FIGURE
    
    ## set up figure grid

    w     = DOUBLE_COLUMN
    goldh = w/2
    fig   = plt.figure(figsize=(w, goldh))
    box_top = 0.94
    box_bot = 0.075
    box_lft = 0.07
    box_rgt = 0.98
    space_y = 0.3
    box_dy  = 0.12
    box_dx  = 0.08
    box_ddx = 0.02

    ha_ratio = 1026 / 1803  # hemagglutinin image size 1026 wide x 1803 high
    ha_dy    = 0.05
    ha_y     = (box_top + ha_dy - box_bot) - space_y - box_dy
    ha_x     = ha_y * ha_ratio * (goldh / w)
    s_x      = (box_rgt - box_lft - (2*ha_x + 5*box_dx/2))/2
    space_x  = (box_rgt - box_lft - 1.2*box_dx - box_ddx)/2

    box_cb   = dict(left=box_lft-box_dx/2+0.02,          right=box_lft+0.02,                        top=box_top-0.05,        bottom=box_top-ha_y+0.05)
    box_h3   = dict(left=box_lft,                        right=box_lft+ha_x,                        top=box_top,             bottom=box_top-ha_y)
    box_h3_s = dict(left=box_h3['right']+box_dx,         right=box_h3['right']+box_dx+s_x,          top=box_top,             bottom=box_top-ha_y+ha_dy)
    box_h1   = dict(left=box_h3['right']+3*box_dx/2+s_x, right=box_h3['right']+3*box_dx/2+s_x+ha_x, top=box_top,             bottom=box_top-ha_y)
    box_h1_s = dict(left=box_h1['right']+box_dx,         right=box_rgt,                             top=box_top,             bottom=box_top-ha_y+ha_dy)
    box_sha  = dict(left=box_lft+box_ddx,                right=box_lft+space_x+box_ddx,             top=box_top-ha_y-box_dy, bottom=box_bot)
    box_sna  = dict(left=box_lft+space_x+1.2*box_dx+box_ddx, right=box_rgt,                             top=box_top-ha_y-box_dy, bottom=box_bot)
    
    ha_hspace   = 0.30

    gs_h3    = gridspec.GridSpec(1, 1, **box_h3)
    ax_h3    = plt.subplot(gs_h3[0, 0])
    gs_h3_s  = gridspec.GridSpec(2, 1, hspace=ha_hspace, **box_h3_s)
    ax_h3_s  = [plt.subplot(gs_h3_s[i, 0]) for i in range(2)]
    gs_h1    = gridspec.GridSpec(1, 1, **box_h1)
    ax_h1    = plt.subplot(gs_h1[0, 0])
    gs_h1_s  = gridspec.GridSpec(2, 1, hspace=ha_hspace, **box_h1_s)
    ax_h1_s  = [plt.subplot(gs_h1_s[i, 0]) for i in range(2)]
    gs_cb    = gridspec.GridSpec(1, 1, **box_cb)
    ax_cb    = plt.subplot(gs_cb[0, 0])
    gs_sha   = gridspec.GridSpec(1, 1, **box_sha)
    ax_sha   = plt.subplot(gs_sha[0, 0])
    gs_sna   = gridspec.GridSpec(1, 1, **box_sna)
    ax_sna   = plt.subplot(gs_sna[0, 0])

    dx_label = -0.03
    dy_label = 0.02

    # a/b -- selection on structure

    # hemagglutinin images
    img_h3 = mpimg.imread('%s/structure/h3_crop.png' % DATA_DIR)
    img_h1 = mpimg.imread('%s/structure/h1_crop.png' % DATA_DIR)
    
    ax_h3.imshow(img_h3)
    ax_h1.imshow(img_h1)

    ax_h3.axis('off')
    ax_h1.axis('off')

    ax_h3.text(box_h3['left']+dx_label, box_h3['top']+dy_label, 'a'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_h1.text(box_h1['left']+dx_label+0.03, box_h1['top']+dy_label, 'b'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_sha.text(box_h3['left']+dx_label, box_sha['top']+dy_label+0.02, 'c'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_sna.text(box_h1['left']+dx_label+0.03, box_sna['top']+dy_label+0.02, 'd'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # colorbar -- rotate by 90*, scale by 3, and translate to middle of axis vertically

    img_cb = mpimg.imread('%s/structure/colorbar_crop.png' % DATA_DIR)
    tr = matplotlib.transforms.Affine2D().rotate_deg(90).scale(1, -3).translate(0, 1.55*img_cb.shape[1]) + ax_cb.transData
    ax_cb.imshow(img_cb, transform=tr, clip_on=False)
    ax_cb.axis('off')

    ax_cb.text(box_cb['left']-0.02, (box_cb['top']+box_cb['bottom'])/2, 'Selection coefficient', transform=fig.transFigure, ha='center', va='center', rotation=90, **DEF_LABELPROPS)
    ax_cb.text(box_cb['left']-0.0085, box_cb['top']-0.06, r'$+$', rotation=90, transform=fig.transFigure, **DEF_LABELPROPS)
    ax_cb.text(box_cb['left']-0.0088, box_cb['bottom']+0.06, r'$-$', rotation=90, transform=fig.transFigure, **DEF_LABELPROPS)

    # a/b -- enrichment in functional sites

    # Read in selection data (mutation level)
    df = [pd.read_csv(s_file, comment='#', memory_map=True) for s_file in s_files_mut]
    df = [d[d['MutationFrequency']>=min_counts] for d in df]
    
    # H3 subsets: RBS, Koel
    subsets = ['RBS', 'Koel']
    for i in range(len(subsets)):
        subset = subsets[i]
        df_h3 = df[0]
        s_sub = df_h3[(df_h3[subset]==True)]['Selection_coefficient']
        s_non = df_h3[(df_h3[subset]==False)]['Selection_coefficient']

        histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.7, edgecolor='none', orientation='vertical')

        pprops = { 'xlim'        : [-0.005, 0.015],
                   'xticks'      : [-0.005,   0.,   0.005,   0.01,   0.015],
                   'xticklabels' : [r'$-0.5$', r'$0$', r'$0.5$', r'$1$', r'$1.5$'],
                   'ylim'        : [0.001, 1],
                   'yticks'      : [0.001, 0.01, 0.1, 1],
                   'bins'        : np.arange(-0.005, 0.015+0.001, 0.0005),
                   'combine'     : True,
                   'plotprops'   : histprops,
                   'axoffset'    : 0.1,
                   'theme'       : 'open' }

        ax_h3_s[i].set_yscale('log')
        if i==len(subsets)-1:
            pprops['xlabel'] = 'Inferred selection coefficient (%)'
        else:
            pprops['xticklabels'] = ['' for k in pprops['xticks']]

        c_plot = [colors[0]]
        mp.hist(             ax=ax_h3_s[i], x=[s_non], colors=[LCOLOR], **pprops)
        mp.plot(type='hist', ax=ax_h3_s[i], x=[s_sub], colors=c_plot, **pprops)

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'right', 'va': 'center', 'clip_on': False}
        tdy = 0.035
        x0 = box_h3_s['right'] + 0.005
        t, p = st.ttest_ind(s_sub, s_non)
        p_exp = np.floor(np.log10(p))
        p_mult = np.round(p / 10**p_exp)
        if subset=='RBS':
            y0 = box_h3_s['top']-0.01
            ax_h3_s[i].text(x0, y0, 'RBS', color=colors[0], fontweight='bold', transform=fig.transFigure, **txtprops)
            ax_h3_s[i].text(x0, y0-tdy, 'other', color=LCOLOR, fontweight='bold', transform=fig.transFigure, **txtprops)
            # ax_h3_s[i].text(x0, y0-2*tdy, r'$p = %d \times 10^{%d}$' % (p_mult, p_exp), color=BKCOLOR, transform=fig.transFigure, **txtprops)
            ax_h3_s[i].text(x0, y0-2*tdy, r'$p < 10^{%d}$' % (p_exp+1), color=BKCOLOR, transform=fig.transFigure, **txtprops)
        elif subset=='Koel':
            y0 = (box_h3_s['top']+box_h3_s['bottom'])/2 - 0.02
            ax_h3_s[i].text(x0, y0, 'Ag (Koel)', color=colors[0], fontweight='bold', transform=fig.transFigure, **txtprops)
            ax_h3_s[i].text(x0, y0-tdy, 'other', color=LCOLOR, fontweight='bold', transform=fig.transFigure, **txtprops)
            # ax_h3_s[i].text(x0, y0-2*tdy, r'$p = %d \times 10^{%d}$' % (p_mult, p_exp), color=BKCOLOR, transform=fig.transFigure, **txtprops)
            ax_h3_s[i].text(x0, y0-2*tdy, r'$p < 10^{%d}$' % (p_exp+1), color=BKCOLOR, transform=fig.transFigure, **txtprops)

    ax_h3_s[0].text(box_h3_s['left']-0.06, (box_h3_s['top']+box_h3_s['bottom'])/2, 'Frequency', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)

    # label RBS
    h3_x_scale = 0.01
    h3_x_offset = -2.5

    label_x = (np.array([0.35, 0.50, 0.65]) + h3_x_offset) * h3_x_scale
    label_y = 10**np.array([0, 0.5, 0.5])

    txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'color': BKCOLOR, 'ha': 'left', 'va': 'center', 'clip_on': False}
    ax_h3_s[0].text(label_x[-1]+0.0005, label_y[-1], 'RBS', **txtprops)
    mp.line(ax=ax_h3_s[0], x=[label_x], y=[label_y], colors=[BKCOLOR], plotprops=dict(lw=AXWIDTH, ls='-', clip_on=False))

    # H1 subset: RBS
    subsets = ['RBS', 'canton_globular_head']
    for i in range(len(subsets)):
        subset = subsets[i]
        df_h1 = df[2]
        s_sub = df_h1[(df_h1[subset]==True)]['Selection_coefficient']
        s_non = df_h1[(df_h1[subset]==False)]['Selection_coefficient']

        histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.7, edgecolor='none', orientation='vertical')

        pprops = { 'xlim'        : [-0.005, 0.015],
                   'xticks'      : [-0.005,   0.,   0.005,   0.01,   0.015],
                   'xticklabels' : [r'$-0.5$', r'$0$', r'$0.5$', r'$1$', r'$1.5$'],
                   'ylim'        : [0.001, 1],
                   'yticks'      : [0.001, 0.01, 0.1, 1],
                   'bins'        : np.arange(-0.005, 0.015+0.001, 0.0005),
                   'combine'     : True,
                   'plotprops'   : histprops,
                   'axoffset'    : 0.1,
                   'theme'       : 'open' }
        
        ax_h1_s[i].set_yscale('log')
        if i==len(subsets)-1:
            pprops['xlabel'] = 'Inferred selection coefficient (%)'
        else:
            pprops['xticklabels'] = ['' for k in pprops['xticks']]
        
        c_plot = [colors[2]]
        mp.hist(             ax=ax_h1_s[i], x=[s_non], colors=[LCOLOR], **pprops)
        mp.plot(type='hist', ax=ax_h1_s[i], x=[s_sub], colors=c_plot, **pprops)

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'right', 'va': 'center', 'clip_on': False}
        tdy = 0.035
        x0 = box_h1_s['right'] + 0.005
        t, p = st.ttest_ind(s_sub, s_non)
        p_exp = np.floor(np.log10(p))
        p_mult = np.round(p / 10**p_exp)
        if subset=='RBS':
            y0 = box_h1_s['top'] - 0.01
            ax_h1_s[i].text(x0, y0, 'RBS', color=colors[2], fontweight='bold', transform=fig.transFigure, **txtprops)
            ax_h1_s[i].text(x0, y0-tdy, 'other', color=LCOLOR, fontweight='bold', transform=fig.transFigure, **txtprops)
            # ax_h1_s[i].text(x0, y0-2*tdy, r'$p = %d \times 10^{%d}$' % (p_mult, p_exp), color=BKCOLOR, transform=fig.transFigure, **txtprops)
            ax_h1_s[i].text(x0, y0-2*tdy, r'$p < 10^{%d}$' % (p_exp+1), color=BKCOLOR, transform=fig.transFigure, **txtprops)
        elif subset=='canton_globular_head':
            y0 = (box_h1_s['top']+box_h1_s['bottom'])/2 - 0.02
            ax_h1_s[i].text(x0, y0, 'Ag (Caton)', color=colors[2], fontweight='bold', transform=fig.transFigure, **txtprops)
            ax_h1_s[i].text(x0, y0-tdy, 'other', color=LCOLOR, fontweight='bold', transform=fig.transFigure, **txtprops)
            # ax_h1_s[i].text(x0, y0-2*tdy, r'$p = %d \times 10^{%d}$' % (p_mult, p_exp), color=BKCOLOR, transform=fig.transFigure, **txtprops)
            ax_h1_s[i].text(x0, y0-2*tdy, r'$p < 10^{%d}$' % (p_exp+1), color=BKCOLOR, transform=fig.transFigure, **txtprops)

    ax_h1_s[0].text(box_h1_s['left']-0.06, (box_h1_s['top']+box_h1_s['bottom'])/2, 'Frequency', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)

    # c -- selection along HA sequence

    # Read in selection data (smoothed across the protein sequence)
    s_column = 'Selection_coeff_smoothed'
    p_column = 'Position'
    df_smooth = [pd.read_csv(s_file, comment='#', memory_map=True) for s_file in s_files_smooth]
    s = [df_smooth[i][s_column] for i in range(len(df_smooth))]
    p = [df_smooth[i][p_column] for i in range(len(df_smooth))]

    sasa_column = 'SASA_mean_smoothed'
    sasa = [df_smooth[i][sasa_column] for i in range(len(df_smooth))]

    lineprops = { 'lw': SIZELINE*2.0, 'linestyle': '-', 'alpha': 0.7 }
    if not colors:
        colors = sns.husl_palette(len(s[0]))

    pprops = { 'xticks':      [],
               'xticklabels': [],
               'xlim':        [0, np.max(p[0])+1],
               'yticks':      [0, 0.4],
               'yminorticks': [0.1, 0.2, 0.3],
               'nudgey':      1.05,
               'xlabel':      'Position along sequence',
               'ylabel':      'Relative strength\nof selection (HA)',
               'plotprops':   lineprops,
               'axoffset':    0.1,
               'hide':        ['bottom'],
               'theme':       'open' }

    mp.line(             ax=ax_sha, x=[p[0]], y=[s[0]], colors=[colors[0]], **pprops)
    mp.plot(type='line', ax=ax_sha, x=[p[0]], y=[s[2]], colors=[colors[2]], **pprops)

    outlineprops = dict(lw=AXWIDTH, ls='-', alpha=1)
    ha1_start = 17
    ha1_end   = 345
    ha2_start = 345
    ha2_end   = 566
    mp.line(ax=ax_sha, x=[[ha1_start+4, ha1_end-4]], y=[[0.37, 0.37]], colors=[BKCOLOR], plotprops=outlineprops)
    mp.line(ax=ax_sha, x=[[ha2_start+4, ha2_end-4]], y=[[0.37, 0.37]], colors=[BKCOLOR], plotprops=outlineprops)

    txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'fontweight': 'bold', 'ha': 'left', 'va': 'center', 'clip_on': False}
    ax_sha.text(box_sha['left']+0.01, box_sha['top']-0.05,  'H3', color=colors[0], transform=fig.transFigure, **txtprops)
    ax_sha.text(box_sha['left']+0.01, box_sha['top']-0.085, 'H1', color=colors[2], transform=fig.transFigure, **txtprops)

    ax_sha.text((ha1_start + ha1_end) / 2, 0.40, 'HA1', ha='center', va='center', **DEF_LABELPROPS)
    ax_sha.text((ha2_start + ha2_end) / 2, 0.40, 'HA2', ha='center', va='center', **DEF_LABELPROPS)

    # analysis -- correlation between selection in H3 and H1 at same positions
    print('Correlation between selection in H3 and H1')
    print('domain\ttype\t\tr\tp')
    for domain in ['full', 'HA1', 'HA2']:
        if domain=='full':
            idxs = np.arange(len(p[0]))
        elif domain=='HA1':
            idxs = np.arange(ha1_start, ha1_end)
        elif domain=='HA2':
            idxs = np.arange(ha2_start, ha2_end)
        for typ in ['pearson', 'spearman']:
            if typ=='pearson':
                r, pval = st.pearsonr(s[0][idxs], s[2][idxs])
            elif typ=='spearman':
                r, pval = st.spearmanr(s[0][idxs], s[2][idxs])
            print('%s\t%s  \t%.3f\t%.2e' % (domain, typ, r, pval))
    print('')

    print('Correlation between SASA in H3 and H1')
    print('domain\ttype\t\tr\tp')
    for domain in ['full', 'HA1', 'HA2']:
        if domain=='full':
            idxs = np.arange(len(p[0]))
        elif domain=='HA1':
            idxs = np.arange(ha1_start, ha1_end)
        elif domain=='HA2':
            idxs = np.arange(ha2_start, ha2_end)
        v1 = np.array([k for k in sasa[0][idxs]])
        v2 = np.array([k for k in sasa[2][idxs]])
        idxs = np.where((v1>0) & (v2>0))[0] # only include positions with nonzero SASA in both H3 and H1
        v1 = v1[idxs]
        v2 = v2[idxs]
        for typ in ['pearson', 'spearman']:
            if typ=='pearson':
                r, pval = st.pearsonr(v1, v2)
            elif typ=='spearman':
                r, pval = st.spearmanr(v1, v2)
            print('%s\t%s  \t%.3f\t%.2e' % (domain, typ, r, pval))
    print('')

    # analysis -- correlation between selection and SASA at same positions
    print('Correlation between selection and SASA in H3')
    print('domain\ttype\t\tr\tp')
    for domain in ['full', 'HA1', 'HA2']:
        if domain=='full':
            idxs = np.arange(len(p[0]))
        elif domain=='HA1':
            idxs = np.arange(ha1_start, ha1_end)
        elif domain=='HA2':
            idxs = np.arange(ha2_start, ha2_end)
        v1 = np.array([k for k in s[0][idxs]])
        v2 = np.array([k for k in sasa[0][idxs]])
        idxs = np.where(v2>0)[0] # only include positions with nonzero SASA
        v1 = v1[idxs]
        v2 = v2[idxs]
        for typ in ['pearson', 'spearman']:
            if typ=='pearson':
                r, pval = st.pearsonr(v1, v2)
            elif typ=='spearman':
                r, pval = st.spearmanr(v1, v2)
            print('%s\t%s  \t%.3f\t%.2e' % (domain, typ, r, pval))
    print('')

    print('Correlation between selection and SASA in H1')
    print('domain\ttype\t\tr\tp')
    for domain in ['full', 'HA1', 'HA2']:
        if domain=='full':
            idxs = np.arange(len(p[2]))
        elif domain=='HA1':
            idxs = np.arange(ha1_start, ha1_end)
        elif domain=='HA2':
            idxs = np.arange(ha2_start, ha2_end)
        v1 = np.array([k for k in s[2][idxs]])
        v2 = np.array([k for k in sasa[2][idxs]])
        idxs = np.where(v2>0)[0] # only include positions with nonzero SASA
        v1 = v1[idxs]
        v2 = v2[idxs]
        for typ in ['pearson', 'spearman']:
            if typ=='pearson':
                r, pval = st.pearsonr(v1, v2)
            elif typ=='spearman':
                r, pval = st.spearmanr(v1, v2)
            print('%s\t%s  \t%.3f\t%.2e' % (domain, typ, r, pval))
    print('')

    # d -- selection along NA sequence

    pprops = { 'xticks':      [],
               'xticklabels': [],
               'xlim':        [0, np.max(p[1])+1],
               'yticks':      [0, 0.2],
               'yminorticks': [0.05, 0.1, 0.15],
               'nudgey':      1.05,
               'xlabel':      'Position along sequence',
               'ylabel':      'Relative strength\nof selection (NA)',
               'plotprops':   lineprops,
               'axoffset':    0.1,
               'hide':        ['bottom'],
               'theme':       'open' }
    
    mp.line(             ax=ax_sna, x=[p[1]], y=[s[1]], colors=[colors[1]], **pprops)
    mp.plot(type='line', ax=ax_sna, x=[p[1]], y=[s[3]], colors=[colors[3]], **pprops)

    ax_sna.text(box_sna['left']+0.01, box_sna['top']-0.01,  'N2', color=colors[1], transform=fig.transFigure, **txtprops)
    ax_sna.text(box_sna['left']+0.01, box_sna['top']-0.045, 'N1', color=colors[3], transform=fig.transFigure, **txtprops)

    # analysis -- correlation between selection in N2 and N1 at same positions
    print('Correlation between selection in N2 and N1')
    print('domain\ttype\t\tr\tp')
    for domain in ['full']:
        if domain=='full':
            idxs = np.arange(len(p[1]))
        for typ in ['pearson', 'spearman']:
            if typ=='pearson':
                r, pval = st.pearsonr(s[1][idxs], s[3][idxs])
            elif typ=='spearman':
                r, pval = st.spearmanr(s[1][idxs], s[3][idxs])
            print('%s\t%s  \t%.3f\t%.2e' % (domain, typ, r, pval))
    print('')

    print('Correlation between SASA in N2 and N1')
    print('domain\ttype\t\tr\tp')
    for domain in ['full']:
        if domain=='full':
            idxs = np.arange(len(p[1]))
        v1 = np.array([k for k in sasa[1][idxs]])
        v2 = np.array([k for k in sasa[3][idxs]])
        idxs = np.where((v1>0) & (v2>0))[0] # only include positions with nonzero SASA
        v1 = v1[idxs]
        v2 = v2[idxs]
        for typ in ['pearson', 'spearman']:
            if typ=='pearson':
                r, pval = st.pearsonr(v1, v2)
            elif typ=='spearman':
                r, pval = st.spearmanr(v1, v2)
            print('%s\t%s  \t%.3f\t%.2e' % (domain, typ, r, pval))
    print('')

    print('Correlation between selection and SASA in N2')
    print('domain\ttype\t\tr\tp')
    for domain in ['full']:
        if domain=='full':
            idxs = np.arange(len(p[1]))
        v1 = np.array([k for k in s[1][idxs]])
        v2 = np.array([k for k in sasa[1][idxs]])
        idxs = np.where(v2>0)[0] # only include positions with nonzero SASA
        v1 = v1[idxs]
        v2 = v2[idxs]
        for typ in ['pearson', 'spearman']:
            if typ=='pearson':
                r, pval = st.pearsonr(v1, v2)
            elif typ=='spearman':
                r, pval = st.spearmanr(v1, v2)
            print('%s\t%s  \t%.3f\t%.2e' % (domain, typ, r, pval))
    print('')

    print('Correlation between selection and SASA in N1')
    print('domain\ttype\t\tr\tp')
    for domain in ['full']:
        if domain=='full':
            idxs = np.arange(len(p[3]))
        v1 = np.array([k for k in s[3][idxs]])
        v2 = np.array([k for k in sasa[3][idxs]])
        idxs = np.where(v2>0)[0] # only include positions with nonzero SASA
        v1 = v1[idxs]
        v2 = v2[idxs]
        for typ in ['pearson', 'spearman']:
            if typ=='pearson':
                r, pval = st.pearsonr(v1, v2)
            elif typ=='spearman':
                r, pval = st.spearmanr(v1, v2)
            print('%s\t%s  \t%.3f\t%.2e' % (domain, typ, r, pval))
    print('')

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_selection_sasa(s_files_smooth, colors=None, fig_name='fig-selection-sasa.pdf', **kwargs):
    ''' Plot smoothed selection across the protein sequence vs. smoothed solvent accessible surface area. '''

    # PLOT FIGURE
    
    ## set up figure grid

    w     = DOUBLE_COLUMN
    goldh = w/2
    fig   = plt.figure(figsize=(w, goldh))
    box_top = 0.96
    box_bot = 0.11
    box_lft = 0.075
    box_rgt = 0.93
    box_dx  = 0.16

    space_x  = (box_rgt - box_lft - box_dx)/2

    box_sha  = dict(left=box_lft,                right=box_lft+space_x, top=box_top, bottom=box_bot)
    box_sna  = dict(left=box_lft+space_x+box_dx, right=box_rgt,         top=box_top, bottom=box_bot)

    gs_sha      = gridspec.GridSpec(2, 1, **box_sha)
    ax_sha      = [plt.subplot(gs_sha[0, 0]), plt.subplot(gs_sha[1, 0])]
    ax_sha_sasa = [ax.twinx() for ax in ax_sha]
    gs_sna      = gridspec.GridSpec(2, 1, **box_sna)
    ax_sna      = [plt.subplot(gs_sna[0, 0]), plt.subplot(gs_sna[1, 0])]
    ax_sna_sasa = [ax.twinx() for ax in ax_sna]

    # a/b -- selection vs. SASA for HA

    # Read in selection data (smoothed across the protein sequence)
    s_column = 'Selection_coeff_smoothed'
    p_column = 'Position'
    sasa_column = 'SASA_mean_smoothed'
    df_smooth = [pd.read_csv(s_file, comment='#', memory_map=True) for s_file in s_files_smooth]
    s = [df_smooth[i][s_column] for i in range(len(df_smooth))]
    p = [df_smooth[i][p_column] for i in range(len(df_smooth))]

    sasa = [df_smooth[i][df_smooth[i][sasa_column]>0][sasa_column] for i in range(len(df_smooth))]
    p_sasa = [df_smooth[i][df_smooth[i][sasa_column]>0][p_column] for i in range(len(df_smooth))]

    lineprops = { 'lw': SIZELINE*2.0, 'linestyle': '-',  'alpha': 0.7 }
    dashprops = { 'lw': SIZELINE*2.0, 'linestyle': '--', 'alpha': 0.7 }
    if not colors:
        colors = sns.husl_palette(len(s[0]))

    pprops_nox = { 'xlim':     [0, 600],
                   'xticks':   [0, 600],
                   'xticklabels': ['', ''],
                   'nudgey':   1.05,
                   'axoffset': 2   }
    pprops_showx = { 'xlim':     [0, 600],
                    'xticks':   [0, 100, 200, 300, 400, 500, 600],
                    'xticklabels': [0, 100, 200, 300, 400, 500, 600],
                    'xlabel':   'Position along sequence',
                    'nudgey':   1.05,
                    'axoffset': 2   }
    pprops_s = { 'ylim': [0, 0.4],
                 'yticks': [0, 0.4],
                 'yminorticks': [0.1, 0.2, 0.3],
                 'ylabel': 'Relative strength of selection',
                 'plotprops': lineprops }
    pprops_sasa = { 'tickprops': DEF_TICKPROPS | dict(right=True, left=False, bottom=False, top=False),
                    'show': ['right'],
                    'hide': ['left', 'bottom', 'top'],
                    'theme': 'none',
                    'ylim': [0, 80],
                    'yticks': [0, 20, 40, 60, 80],
                    'ylabel': 'Solvent accessible surface area',
                    'plotprops': dashprops }

    mp.plot(type='line', ax=ax_sha[0], x=[p[0]], y=[s[0]], colors=[colors[0]], **(pprops_nox | pprops_s))
    mp.plot(type='line', ax=ax_sha_sasa[0], x=[p_sasa[0]], y=[sasa[0]], colors=[LCOLOR], **(pprops_nox | pprops_sasa))
    mp.plot(type='line', ax=ax_sha[1], x=[p[2]], y=[s[2]], colors=[colors[2]], **(pprops_showx | pprops_s))
    mp.plot(type='line', ax=ax_sha_sasa[1], x=[p_sasa[2]], y=[sasa[2]], colors=[LCOLOR], **(pprops_showx | pprops_sasa))
    

    outlineprops = dict(lw=AXWIDTH, ls='-', alpha=1)
    ha1_start = 17
    ha1_end   = 345
    ha2_start = 345
    ha2_end   = 566
    mp.line(ax=ax_sha[0], x=[[ha1_start+4, ha1_end-4]], y=[[0.37, 0.37]], colors=[BKCOLOR], plotprops=outlineprops)
    mp.line(ax=ax_sha[0], x=[[ha2_start+4, ha2_end-4]], y=[[0.37, 0.37]], colors=[BKCOLOR], plotprops=outlineprops)

    mp.line(ax=ax_sha[1], x=[[ha1_start+4, ha1_end-4]], y=[[0.37, 0.37]], colors=[BKCOLOR], plotprops=outlineprops)
    mp.line(ax=ax_sha[1], x=[[ha2_start+4, ha2_end-4]], y=[[0.37, 0.37]], colors=[BKCOLOR], plotprops=outlineprops)

    txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'fontweight': 'bold', 'ha': 'left', 'va': 'center', 'clip_on': False}
    ax_sha[0].text(box_sha['left']+0.005, box_sha['top']-0.07, 'H3 selection', color=colors[0], transform=fig.transFigure, **txtprops)
    ax_sha[0].text(box_sha['left']+0.005, box_sha['top']-0.10, 'H3 SASA', color=LCOLOR, transform=fig.transFigure, **txtprops)
    ax_sha[1].text(box_sha['left']+0.005, (box_sha['top']+box_sha['bottom'])/2-0.13, 'H1 selection', color=colors[2], transform=fig.transFigure, **txtprops)
    ax_sha[1].text(box_sha['left']+0.005, (box_sha['top']+box_sha['bottom'])/2-0.16, 'H1 SASA', color=LCOLOR, transform=fig.transFigure, **txtprops)

    ax_sha[0].text((ha1_start + ha1_end) / 2, 0.40, 'HA1', ha='center', va='center', **DEF_LABELPROPS)
    ax_sha[0].text((ha2_start + ha2_end) / 2, 0.40, 'HA2', ha='center', va='center', **DEF_LABELPROPS)

    # c/d -- selection vs. SASA for NA

    pprops_nox = { 'xlim':     [0, 500],
                   'xticks':   [0, 500],
                   'xticklabels': ['', ''],
                   'nudgey':   1.05,
                   'axoffset': 2   }
    pprops_showx = { 'xlim':     [0, 500],
                    'xticks':   [0, 100, 200, 300, 400, 500],
                    'xticklabels': [0, 100, 200, 300, 400, 500],
                    'xlabel':   'Position along sequence',
                    'nudgey':   1.05,
                    'axoffset': 2   }
    pprops_s = { 'ylim': [0, 0.2],
                 'yticks': [0, 0.2],
                 'yminorticks': [0.05, 0.1, 0.15],
                 'ylabel': 'Relative strength of selection',
                 'plotprops': lineprops }
    pprops_sasa = { 'tickprops': DEF_TICKPROPS | dict(right=True, left=False, bottom=False, top=False),
                    'show': ['right'],
                    'hide': ['left', 'bottom', 'top'],
                    'theme': 'none',
                    'ylim': [0, 80],
                    'yticks': [0, 20, 40, 60, 80],
                    'ylabel': 'Solvent accessible surface area',
                    'plotprops': dashprops }

    mp.plot(type='line', ax=ax_sna[0], x=[p[1]], y=[s[1]], colors=[colors[1]], **(pprops_nox | pprops_s))
    mp.plot(type='line', ax=ax_sna_sasa[0], x=[p_sasa[1]], y=[sasa[1]], colors=[LCOLOR], **(pprops_nox | pprops_sasa))
    mp.plot(type='line', ax=ax_sna[1], x=[p[3]], y=[s[3]], colors=[colors[3]], **(pprops_showx | pprops_s))
    mp.plot(type='line', ax=ax_sna_sasa[1], x=[p_sasa[3]], y=[sasa[3]], colors=[LCOLOR], **(pprops_showx | pprops_sasa))

    ax_sna[0].text(box_sna['left']+0.005, box_sna['top']-0.01, 'N2 selection', color=colors[1], transform=fig.transFigure, **txtprops)
    ax_sna[0].text(box_sna['left']+0.005, box_sna['top']-0.04, 'N2 SASA', color=LCOLOR, transform=fig.transFigure, **txtprops)
    ax_sna[1].text(box_sna['left']+0.005, (box_sna['top']+box_sna['bottom'])/2-0.01, 'N1 selection', color=colors[3], transform=fig.transFigure, **txtprops)
    ax_sna[1].text(box_sna['left']+0.005, (box_sna['top']+box_sna['bottom'])/2-0.04, 'N1 SASA', color=LCOLOR, transform=fig.transFigure, **txtprops)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_frequency_selection_structure(freq_files, s_files, deltats=[], min_counts=2, colors=None, fig_name='fig-frequency-selection-structure.pdf'):
    """ Plot mutation frequency trajectories, distribution of selection coefficients,
        selection on HA structure, and enrichment in beneficial mutations at functional sites. """

    # Read in frequency data
    x = [np.loadtxt(f) for f in freq_files]
    for i in range(len(x)):
        above_cut = np.array([np.max(x[i][j])>0.01 for j in range(len(x[i]))])
        x[i] = x[i][above_cut]
        not_WT = np.array([x[i][j][0]<0.95 for j in range(len(x[i]))])
        x[i] = x[i][not_WT]
    t = [np.arange(0, len(x[i][0])*10, 10) for i in range(len(x))]

    if not deltats:
        deltats = [0 for i in range(len(x))]

    # Read in selection data
    df = [pd.read_csv(s_file, comment='#', memory_map=True) for s_file in s_files]
    df = [d[d['MutationFrequency']>=min_counts] for d in df]
    s_all = [d['Selection_coefficient'] for d in df]

    # PLOT FIGURE
    
    ## set up figure grid

    w     = DOUBLE_COLUMN
    goldh = w/1.5
    fig   = plt.figure(figsize=(w, goldh))
    box_top = 0.95
    box_bot = 0.075
    box_lft = 0.07
    traj_y  = 0.45
    box_dy  = 0.12
    box_dx  = 0.08

    ha_ratio = 1026 / 1803  # hemagglutinin image size 1026 wide x 1803 high
    ha_dy    = 0.05
    ha_y     = (box_top + ha_dy - box_bot) - traj_y - box_dy
    ha_x     = ha_y * ha_ratio * (goldh / w)

    box_traj = dict(left=box_lft,                  right=0.65,                    top=box_top,               bottom=box_top-traj_y)
    box_sel  = dict(left=box_traj['right']+box_dx, right=0.95,                    top=box_top,               bottom=box_top-traj_y)
    box_h3   = dict(left=box_lft+box_dx/2,         right=box_lft+(box_dx/2)+ha_x, top=box_top-traj_y-box_dy, bottom=box_bot-ha_dy)
    box_h3_s = dict(left=box_h3['right']+box_dx,   right=0.5-(box_dx/4),          top=box_top-traj_y-box_dy, bottom=box_bot)
    box_h1   = dict(left=0.5+(box_dx/4),           right=0.5+(box_dx/4)+ha_x,     top=box_top-traj_y-box_dy, bottom=box_bot-ha_dy)
    box_h1_s = dict(left=box_h1['right']+box_dx,   right=0.95,                    top=box_top-traj_y-box_dy, bottom=box_bot)
    box_cb   = dict(left=box_lft+0.01,             right=box_lft+(box_dx/2)+0.01, top=box_top-traj_y-box_dy-0.05, bottom=box_bot-ha_dy+0.05)

    traj_hspace = 0.40
    ha_hspace   = 0.30

    gs_traj  = gridspec.GridSpec(4, 1, hspace=traj_hspace, **box_traj)
    ax_traj  = [plt.subplot(gs_traj[i, 0]) for i in range(4)]
    gs_sel   = gridspec.GridSpec(4, 1, hspace=traj_hspace, **box_sel)
    ax_sel   = [plt.subplot(gs_sel[i, 0]) for i in range(4)]
    gs_h3    = gridspec.GridSpec(1, 1, **box_h3)
    ax_h3    = plt.subplot(gs_h3[0, 0])
    gs_h3_s  = gridspec.GridSpec(2, 1, hspace=ha_hspace, **box_h3_s)
    ax_h3_s  = [plt.subplot(gs_h3_s[i, 0]) for i in range(2)]
    gs_h1    = gridspec.GridSpec(1, 1, **box_h1)
    ax_h1    = plt.subplot(gs_h1[0, 0])
    gs_h1_s  = gridspec.GridSpec(2, 1, hspace=ha_hspace, **box_h1_s)
    ax_h1_s  = [plt.subplot(gs_h1_s[i, 0]) for i in range(2)]
    gs_cb    = gridspec.GridSpec(1, 1, **box_cb)
    ax_cb    = plt.subplot(gs_cb[0, 0])

    dx_label = -0.03
    dy_label = 0.02

    # a -- trajectories

    lineprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 0.7 }
    if not colors:
        colors = sns.husl_palette(len(x))

    # times t are in days since Jan 1, 2009; convert to month/year for plot using datetime
    # add x axis label every 6 months

    t_start = dt.datetime(2009, 1, 1)
    t_days  = [i*2*365 for i in range(6)]
    t_dates = [t_start + dt.timedelta(days=int(i)) for i in t_days]
    t_labels = [t_date.strftime('%b %Y') for t_date in t_dates]

    for i in range(len(x)):
        pprops = { 'xticks'      : t_days + [4100],
                   'xticklabels' : ['' for k in t_labels] + [''],
                   'xlim'        : [0, 4100],
                   'yticks'      : [0, 1],
                   'yminorticks' : [0.25, 0.5, 0.75],
                   'nudgey'      : 1.05,
                #    'ylabel'      : 'Allele\nfrequency',
                   'plotprops'   : lineprops,
                   'axoffset'    : 0.1,
                   'theme'       : 'open' }

        xdat = [t[i] + deltats[i] for k in range(len(x[i]))]
        ydat = [k for k in x[i]]
        if i==len(x)-1:
            pprops['xticklabels'] = t_labels + ['']

        mp.plot(type='line', ax=ax_traj[i], x=xdat, y=ydat, colors=[colors[i] for k in range(len(xdat))], **pprops)

    ax_traj[0].text(box_traj['left']-0.04, (box_traj['top']+box_traj['bottom'])/2, 'Allele frequency', transform=fig.transFigure, ha='center', va='center', rotation=90, **DEF_LABELPROPS)
    ax_traj[0].text(box_traj['left']+dx_label, box_traj['top']+dy_label, 'a'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # b -- selection coefficients

    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=1, edgecolor='none', orientation='vertical')
    
    for i in range(len(s_all)):
        pprops = { 'xlim'        : [-0.005, 0.015],
                   'xticks'      : [  -0.005,   0.,   0.005,   0.01,   0.015],
                   'xticklabels' : [r'$-0.5$', r'$0$', r'$0.5$', r'$1$', r'$1.5$'],
                   'ylim'        : [0.001, 1],
                   'yticks'      : [0.001, 0.01, 0.1, 1],
                #    'ylabel'      : 'Frequency',
                   'bins'        : np.arange(-0.005, 0.015+0.001, 0.0005),
                   'combine'     : True,
                   'plotprops'   : histprops,
                   'axoffset'    : 0.1,
                   'theme'       : 'open' }

        ax_sel[i].set_yscale('log')
        if i==len(s_all)-1:
            pprops['xlabel'] = 'Inferred selection coefficient (%)'
        else:
            pprops['xticklabels'] = ['' for k in pprops['xticks']]
        
        mp.plot(type='hist', ax=ax_sel[i], x=[s_all[i]], colors=[colors[i]], **pprops)

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'fontweight': 'bold', 'ha': 'right', 'va': 'center', 'clip_on': False}
        labels = ['H3', 'N2', 'H1', 'N1']
        ax_sel[i].text(box_sel['right']+0.005, box_sel['top']-(box_sel['top']-box_sel['bottom']+0.03)*(i/4), '%s' % labels[i], color=colors[i], transform=fig.transFigure, **txtprops)

        # s_vals = s_all[i].values
        # max_s = np.max(s_vals)
        # idx_max = np.argmax(s_vals)
        # print(labels[i], df[i].iloc[idx_max])

    ax_sel[0].text(box_sel['left']-0.06, (box_sel['top']+box_sel['bottom'])/2, 'Frequency', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)
    ax_sel[0].text(box_sel['left']+dx_label-0.02, box_sel['top']+dy_label, 'b'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # c -- selection on structure

    # hemagglutinin images
    img_h3 = mpimg.imread('%s/structure/h3_crop.png' % DATA_DIR)
    img_h1 = mpimg.imread('%s/structure/h1_crop.png' % DATA_DIR)
    
    ax_h3.imshow(img_h3)
    ax_h1.imshow(img_h1)

    ax_h3.axis('off')
    ax_h1.axis('off')

    ax_h3.text(box_traj['left']+dx_label, box_h3['top']+dy_label, 'c'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_h1.text(box_h1['left']+dx_label+0.03, box_h1['top']+dy_label, 'd'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # colorbar -- rotate by 90*, scale by 3, and translate to middle of axis vertically

    img_cb = mpimg.imread('%s/structure/colorbar_crop.png' % DATA_DIR)
    tr = matplotlib.transforms.Affine2D().rotate_deg(90).scale(1, -3).translate(0, 1.55*img_cb.shape[1]) + ax_cb.transData
    ax_cb.imshow(img_cb, transform=tr, clip_on=False)
    ax_cb.axis('off')

    ax_cb.text(box_cb['left']-0.02, (box_cb['top']+box_cb['bottom'])/2, 'Selection coefficient', transform=fig.transFigure, ha='center', va='center', rotation=90, **DEF_LABELPROPS)
    ax_cb.text(box_cb['left']-0.0085, box_cb['top']-0.025, r'$+$', rotation=90, transform=fig.transFigure, **DEF_LABELPROPS)
    ax_cb.text(box_cb['left']-0.0088, box_cb['bottom']+0.025, r'$-$', rotation=90, transform=fig.transFigure, **DEF_LABELPROPS)

    # d -- enrichment in functional sites
    
    # H3 subsets: RBS, Koel
    subsets = ['RBS', 'Koel']
    for i in range(len(subsets)):
        subset = subsets[i]
        df_h3 = df[0]
        s_sub = df_h3[(df_h3[subset]==True)]['Selection_coefficient']
        s_non = df_h3[(df_h3[subset]==False)]['Selection_coefficient']

        histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.7, edgecolor='none', orientation='vertical')

        pprops = { 'xlim'        : [-0.005, 0.015],
                   'xticks'      : [-0.005,   0.,   0.005,   0.01,   0.015],
                   'xticklabels' : [r'$-0.5$', r'$0$', r'$0.5$', r'$1$', r'$1.5$'],
                   'ylim'        : [0.001, 1],
                   'yticks'      : [0.001, 0.01, 0.1, 1],
                #    'ylabel'      : 'Frequency',
                   'bins'        : np.arange(-0.005, 0.015+0.001, 0.0005),
                   'combine'     : True,
                   'plotprops'   : histprops,
                   'axoffset'    : 0.1,
                   'theme'       : 'open' }

        ax_h3_s[i].set_yscale('log')
        if i==len(subsets)-1:
            pprops['xlabel'] = 'Inferred selection coefficient (%)'
        else:
            pprops['xticklabels'] = ['' for k in pprops['xticks']]

        c_plot = [colors[0]]
        mp.hist(             ax=ax_h3_s[i], x=[s_non], colors=[LCOLOR], **pprops)
        mp.plot(type='hist', ax=ax_h3_s[i], x=[s_sub], colors=c_plot, **pprops)

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'right', 'va': 'center', 'clip_on': False}
        tdy = 0.025
        x0 = box_h3_s['right'] + 0.005
        t, p = st.ttest_ind(s_sub, s_non)
        p_exp = np.floor(np.log10(p))
        p_mult = np.round(p / 10**p_exp)
        if subset=='RBS':
            y0 = box_h3_s['top']
            ax_h3_s[i].text(x0, y0, 'RBS', color=colors[0], fontweight='bold', transform=fig.transFigure, **txtprops)
            ax_h3_s[i].text(x0, y0-tdy, 'other', color=LCOLOR, fontweight='bold', transform=fig.transFigure, **txtprops)
            # ax_h3_s[i].text(x0, y0-2*tdy, r'$p = %d \times 10^{%d}$' % (p_mult, p_exp), color=BKCOLOR, transform=fig.transFigure, **txtprops)
            ax_h3_s[i].text(x0, y0-2*tdy, r'$p < 10^{%d}$' % (p_exp+1), color=BKCOLOR, transform=fig.transFigure, **txtprops)
        elif subset=='Koel':
            y0 = (box_h3_s['top']+box_h3_s['bottom'])/2 - 0.01
            ax_h3_s[i].text(x0, y0, 'Ag (Koel)', color=colors[0], fontweight='bold', transform=fig.transFigure, **txtprops)
            ax_h3_s[i].text(x0, y0-tdy, 'other', color=LCOLOR, fontweight='bold', transform=fig.transFigure, **txtprops)
            # ax_h3_s[i].text(x0, y0-2*tdy, r'$p = %d \times 10^{%d}$' % (p_mult, p_exp), color=BKCOLOR, transform=fig.transFigure, **txtprops)
            ax_h3_s[i].text(x0, y0-2*tdy, r'$p < 10^{%d}$' % (p_exp+1), color=BKCOLOR, transform=fig.transFigure, **txtprops)

    ax_h3_s[0].text(box_h3_s['left']-0.06, (box_h3_s['top']+box_h3_s['bottom'])/2, 'Frequency', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)

    # label RBS
    h3_x_scale = 0.01
    h3_x_offset = -2.5

    label_x = (np.array([0.35, 0.50, 0.65]) + h3_x_offset) * h3_x_scale
    label_y = 10**np.array([0, 0.5, 0.5])

    txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'color': BKCOLOR, 'ha': 'left', 'va': 'center', 'clip_on': False}
    ax_h3_s[0].text(label_x[-1]+0.0005, label_y[-1], 'RBS', **txtprops)
    mp.line(ax=ax_h3_s[0], x=[label_x], y=[label_y], colors=[BKCOLOR], plotprops=dict(lw=AXWIDTH, ls='-', clip_on=False))

    # H1 subset: RBS
    subsets = ['RBS', 'canton_globular_head']
    for i in range(len(subsets)):
        subset = subsets[i]
        df_h1 = df[2]
        s_sub = df_h1[(df_h1[subset]==True)]['Selection_coefficient']
        s_non = df_h1[(df_h1[subset]==False)]['Selection_coefficient']

        histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.7, edgecolor='none', orientation='vertical')

        pprops = { 'xlim'        : [-0.005, 0.015],
                   'xticks'      : [-0.005,   0.,   0.005,   0.01,   0.015],
                   'xticklabels' : [r'$-0.5$', r'$0$', r'$0.5$', r'$1$', r'$1.5$'],
                   'ylim'        : [0.001, 1],
                   'yticks'      : [0.001, 0.01, 0.1, 1],
                #    'ylabel'      : 'Frequency',
                   'bins'        : np.arange(-0.005, 0.015+0.001, 0.0005),
                   'combine'     : True,
                   'plotprops'   : histprops,
                   'axoffset'    : 0.1,
                   'theme'       : 'open' }
        
        ax_h1_s[i].set_yscale('log')
        if i==len(subsets)-1:
            pprops['xlabel'] = 'Inferred selection coefficient (%)'
        else:
            pprops['xticklabels'] = ['' for k in pprops['xticks']]
        
        c_plot = [colors[2]]
        mp.hist(             ax=ax_h1_s[i], x=[s_non], colors=[LCOLOR], **pprops)
        mp.plot(type='hist', ax=ax_h1_s[i], x=[s_sub], colors=c_plot, **pprops)

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'right', 'va': 'center', 'clip_on': False}
        tdy = 0.025
        x0 = box_h1_s['right'] + 0.005
        t, p = st.ttest_ind(s_sub, s_non)
        p_exp = np.floor(np.log10(p))
        p_mult = np.round(p / 10**p_exp)
        if subset=='RBS':
            y0 = box_h1_s['top']
            ax_h1_s[i].text(x0, y0, 'RBS', color=colors[2], fontweight='bold', transform=fig.transFigure, **txtprops)
            ax_h1_s[i].text(x0, y0-tdy, 'other', color=LCOLOR, fontweight='bold', transform=fig.transFigure, **txtprops)
            # ax_h1_s[i].text(x0, y0-2*tdy, r'$p = %d \times 10^{%d}$' % (p_mult, p_exp), color=BKCOLOR, transform=fig.transFigure, **txtprops)
            ax_h1_s[i].text(x0, y0-2*tdy, r'$p < 10^{%d}$' % (p_exp+1), color=BKCOLOR, transform=fig.transFigure, **txtprops)
        elif subset=='canton_globular_head':
            y0 = (box_h1_s['top']+box_h1_s['bottom'])/2 - 0.01
            ax_h1_s[i].text(x0, y0, 'Ag (Caton)', color=colors[2], fontweight='bold', transform=fig.transFigure, **txtprops)
            ax_h1_s[i].text(x0, y0-tdy, 'other', color=LCOLOR, fontweight='bold', transform=fig.transFigure, **txtprops)
            # ax_h1_s[i].text(x0, y0-2*tdy, r'$p = %d \times 10^{%d}$' % (p_mult, p_exp), color=BKCOLOR, transform=fig.transFigure, **txtprops)
            ax_h1_s[i].text(x0, y0-2*tdy, r'$p < 10^{%d}$' % (p_exp+1), color=BKCOLOR, transform=fig.transFigure, **txtprops)

    ax_h1_s[0].text(box_h1_s['left']-0.06, (box_h1_s['top']+box_h1_s['bottom'])/2, 'Frequency', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_frequency_selection(freq_files, s_files, deltats=[], min_counts=2, colors=None, fig_name='fig-frequency-selection.pdf'):
    """ Plot mutation frequency trajectories and distribution of selection coefficients. """

    # Read in frequency data
    x = [np.loadtxt(f) for f in freq_files]
    for i in range(len(x)):
        above_cut = np.array([np.max(x[i][j])>0.01 for j in range(len(x[i]))])
        x[i] = x[i][above_cut]
        not_WT = np.array([x[i][j][0]<0.95 for j in range(len(x[i]))])
        x[i] = x[i][not_WT]
    t = [np.arange(0, len(x[i][0])*10, 10) for i in range(len(x))]

    if not deltats:
        deltats = [0 for i in range(len(x))]

    # Read in selection data
    df = [pd.read_csv(s_file, comment='#', memory_map=True) for s_file in s_files]
    df = [d[d['MutationFrequency']>=min_counts] for d in df]
    s_all = [d['Selection_coefficient'] for d in df]

    # PLOT FIGURE
    
    ## set up figure grid

    w     = DOUBLE_COLUMN
    goldh = w/1.8
    fig   = plt.figure(figsize=(w, goldh))
    box_top = 0.95
    box_bot = 0.10
    box_lft = 0.07
    box_dx  = 0.08

    box_traj = dict(left=box_lft,                  right=0.65, top=box_top, bottom=box_bot)
    box_sel  = dict(left=box_traj['right']+box_dx, right=0.95, top=box_top, bottom=box_bot)
    
    traj_hspace = 0.40

    gs_traj  = gridspec.GridSpec(4, 1, hspace=traj_hspace, **box_traj)
    ax_traj  = [plt.subplot(gs_traj[i, 0]) for i in range(4)]
    gs_sel   = gridspec.GridSpec(4, 1, hspace=traj_hspace, **box_sel)
    ax_sel   = [plt.subplot(gs_sel[i, 0]) for i in range(4)]

    dx_label = -0.03
    dy_label = 0.02

    # a -- trajectories

    lineprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 0.7 }
    if not colors:
        colors = sns.husl_palette(len(x))

    # times t are in days since Jan 1, 2009; convert to month/year for plot using datetime
    # add x axis label every 6 months

    t_start = dt.datetime(2009, 1, 1)
    t_days  = [i*2*365 for i in range(6)]
    t_dates = [t_start + dt.timedelta(days=int(i)) for i in t_days]
    t_labels = [t_date.strftime('%b %Y') for t_date in t_dates]

    for i in range(len(x)):
        pprops = { 'xticks'      : t_days + [4100],
                   'xticklabels' : ['' for k in t_labels] + [''],
                   'xlim'        : [0, 4100],
                   'yticks'      : [0, 1],
                   'yminorticks' : [0.25, 0.5, 0.75],
                   'nudgey'      : 1.05,
                #    'ylabel'      : 'Allele\nfrequency',
                   'plotprops'   : lineprops,
                   'axoffset'    : 0.1,
                   'theme'       : 'open' }

        xdat = [t[i] + deltats[i] for k in range(len(x[i]))]
        ydat = [k for k in x[i]]
        if i==len(x)-1:
            pprops['xticklabels'] = t_labels + ['']

        mp.plot(type='line', ax=ax_traj[i], x=xdat, y=ydat, colors=[colors[i] for k in range(len(xdat))], **pprops)

    ax_traj[0].text(box_traj['left']-0.04, (box_traj['top']+box_traj['bottom'])/2, 'Allele frequency', transform=fig.transFigure, ha='center', va='center', rotation=90, **DEF_LABELPROPS)
    ax_traj[0].text(box_traj['left']+dx_label, box_traj['top']+dy_label, 'a'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # b -- selection coefficients

    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=1, edgecolor='none', orientation='vertical')
    
    for i in range(len(s_all)):
        pprops = { 'xlim'        : [-0.005, 0.015],
                   'xticks'      : [  -0.005,   0.,   0.005,   0.01,   0.015],
                   'xticklabels' : [r'$-0.5$', r'$0$', r'$0.5$', r'$1$', r'$1.5$'],
                   'ylim'        : [0.001, 1],
                   'yticks'      : [0.001, 0.01, 0.1, 1],
                #    'ylabel'      : 'Frequency',
                   'bins'        : np.arange(-0.005, 0.015+0.001, 0.0005),
                   'combine'     : True,
                   'plotprops'   : histprops,
                   'axoffset'    : 0.1,
                   'theme'       : 'open' }

        ax_sel[i].set_yscale('log')
        if i==len(s_all)-1:
            pprops['xlabel'] = 'Inferred selection coefficient (%)'
        else:
            pprops['xticklabels'] = ['' for k in pprops['xticks']]
        
        mp.plot(type='hist', ax=ax_sel[i], x=[s_all[i]], colors=[colors[i]], **pprops)

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'fontweight': 'bold', 'ha': 'right', 'va': 'center', 'clip_on': False}
        labels = ['H3', 'N2', 'H1', 'N1']
        ax_sel[i].text(box_sel['right']+0.005, box_sel['top']-(box_sel['top']-box_sel['bottom']+0.08)*(i/4), '%s' % labels[i], color=colors[i], transform=fig.transFigure, **txtprops)

        # s_vals = s_all[i].values
        # max_s = np.max(s_vals)
        # idx_max = np.argmax(s_vals)
        # print(labels[i], df[i].iloc[idx_max])

    ax_sel[0].text(box_sel['left']-0.06, (box_sel['top']+box_sel['bottom'])/2, 'Frequency', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)
    ax_sel[0].text(box_sel['left']+dx_label-0.02, box_sel['top']+dy_label, 'b'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_selection_space_time(s_files, dsdt_files, t_files, colors=None, fig_name='fig-selection-space-time.pdf'):
# def plot_selection_space(s_files, colors=None, fig_name='fig-selection-space.pdf'):
    """ Plot selection across HA and NA in space (along the protein sequence). """ # and over time. """

    # Read in selection data
    s_columns = [['s_h3', 's_h1'], ['s_N2', 's_N1']]
    p_columns = ['position', 'position']
    df = [pd.read_csv(s_file, comment='#', memory_map=True) for s_file in s_files]
    s = [[df[i][c] for c in s_columns[i]] for i in range(len(df))]
    p = [df[i][c] for i, c in enumerate(p_columns)]

    # Read in selection over time data
    t_0_dates = [dt.datetime(2009, 1, 1), dt.datetime(2009, 1, 1)]
    t = []
    for i, t_file in enumerate(t_files):
        t_dates = []
        with open(t_file, 'r') as f:
            for line in f:
                date_str = line.strip()
                date_obj = dt.datetime.strptime(date_str, '%Y-%m-%d')
                t_dates.append(date_obj)
        t_days = [(date - t_0_dates[i]).days for date in t_dates]
        t.append(t_days)
    dsdt = [np.loadtxt(dsdt_file) for dsdt_file in dsdt_files]

    # Sum selection over time across all sites
    s_time = []
    for i in range(len(dsdt)):
        s_sum = np.sum(dsdt[i], axis=1)
        s_time.append(s_sum)
    
    # Compute cross-correlation between overall change in selection over time between H3 and H1 (first two entries in s_time)
    corr = np.correlate(s_time[0] - np.mean(s_time[0]), s_time[1] - np.mean(s_time[1]), mode='full')

    # PLOT FIGURE
    
    ## set up figure grid

    w     = SINGLE_COLUMN
    goldh = w/1.5
    fig   = plt.figure(figsize=(w, goldh))
    box_top = 0.94
    box_bot = 0.08
    box_lft = 0.16
    box_rgt = 0.95 
    box_dy  = 0.12
    # box_dx  = 0.08
    pos_x   = box_rgt - box_lft  # 0.40
    pos_y   = (box_top - box_bot - box_dy) / 2

    box_sha = dict(left=box_lft, right=box_lft+pos_x, top=box_top,              bottom=box_top-pos_y)
    box_sna = dict(left=box_lft, right=box_lft+pos_x, top=box_top-pos_y-box_dy, bottom=box_bot)
    # box_tha = dict(left=box_sha['right']+box_dx, right=box_rgt,       top=box_top,              bottom=box_top-pos_y)
    # box_tna = dict(left=box_sna['right']+box_dx, right=box_rgt,       top=box_top-pos_y-box_dy, bottom=box_bot)

    # p_hspace = 0.40

    gs_sha  = gridspec.GridSpec(1, 1, **box_sha)
    ax_sha  = plt.subplot(gs_sha[0, 0])
    gs_sna  = gridspec.GridSpec(1, 1, **box_sna)
    ax_sna  = plt.subplot(gs_sna[0, 0])
    # gs_tha  = gridspec.GridSpec(1, 2, hspace=p_hspace, **box_tha)
    # ax_tha  = [plt.subplot(gs_tha[0, i]) for i in range(2)]
    # gs_tna  = gridspec.GridSpec(1, 2, hspace=p_hspace, **box_tna)
    # ax_tna  = [plt.subplot(gs_tna[0, i]) for i in range(2)]

    dx_label = -0.13
    dy_label = 0.02

    # a -- selection along HA sequence

    lineprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 0.7 }
    if not colors:
        colors = sns.husl_palette(len(s[0]))

    pprops = { 'xticks':      [],
               'xticklabels': [],
               'xlim':        [0, np.max(p[0])+1],
               'yticks':      [0, 0.4],
               'yminorticks': [0.1, 0.2, 0.3],
               'nudgey':      1.05,
               'ylabel':      'Relative\nselection in HA',
               'plotprops':   lineprops,
               'axoffset':    0.1,
               'hide':        ['bottom'],
               'theme':       'open' }

    # # plot SASA
    # df_sasa = pd.read_csv(sasa_file, comment='#', memory_map=True)
    # x_sasa = df_sasa['Position']
    # y_sasa = 0.4 * df_sasa['SASA'] / np.max(df_sasa['SASA'])
    # # compute moving average of SASA with window size 10
    # window_size = 10
    # y_sasa_ma = np.convolve(y_sasa, np.ones(window_size)/window_size, mode='same')
    # mp.line(             ax=ax_sha, x=[x_sasa], y=[y_sasa_ma], colors=[LCOLOR], plotprops=dict(lw=AXWIDTH, ls='-', alpha=0.5))

    mp.line(             ax=ax_sha, x=[p[0]], y=[s[0][0]], colors=[colors[0][0]], **pprops)
    mp.plot(type='line', ax=ax_sha, x=[p[0]], y=[s[0][1]], colors=[colors[0][1]], **pprops)

    outlineprops = dict(lw=AXWIDTH, ls='-', alpha=1)
    ha1_start = df[0][df[0]['domain']=='HA1']['position'].min()
    ha1_end   = df[0][df[0]['domain']=='HA1']['position'].max()
    ha2_start = df[0][df[0]['domain']=='HA2']['position'].min()
    ha2_end   = df[0][df[0]['domain']=='HA2']['position'].max()
    mp.line(ax=ax_sha, x=[[ha1_start+4, ha1_end-4]], y=[[0.37, 0.37]], colors=[BKCOLOR], plotprops=outlineprops)
    mp.line(ax=ax_sha, x=[[ha2_start+4, ha2_end-4]], y=[[0.37, 0.37]], colors=[BKCOLOR], plotprops=outlineprops)

    txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'fontweight': 'bold', 'ha': 'left', 'va': 'center', 'clip_on': False}
    ax_sha.text(box_sha['left']+0.01, box_sha['top']-0.08, 'H3', color=colors[0][0], transform=fig.transFigure, **txtprops)
    ax_sha.text(box_sha['left']+0.01, box_sha['top']-0.13, 'H1', color=colors[0][1], transform=fig.transFigure, **txtprops)

    ax_sha.text((ha1_start + ha1_end) / 2, 0.40, 'HA1', ha='center', va='center', **DEF_LABELPROPS)
    ax_sha.text((ha2_start + ha2_end) / 2, 0.40, 'HA2', ha='center', va='center', **DEF_LABELPROPS)

    # analysis -- correlation between selection in H3 and H1 at same positions
    print('Correlation between selection in H3 and H1')
    print('domain\ttype\t\tr\tp')
    for domain in ['full', 'HA1', 'HA2']:
        if domain=='full':
            idxs = np.arange(len(p[0]))
        elif domain=='HA1':
            idxs = df[0][(df[0]['domain']=='HA1')]['position'] - 1
        elif domain=='HA2':
            idxs = df[0][(df[0]['domain']=='HA2')]['position'] - 1
        for typ in ['pearson', 'spearman']:
            if typ=='pearson':
                r, pval = st.pearsonr(s[0][0][idxs], s[0][1][idxs])
            elif typ=='spearman':
                r, pval = st.spearmanr(s[0][0][idxs], s[0][1][idxs])
            print('%s\t%s  \t%.3f\t%.2e' % (domain, typ, r, pval))
    print('')

    # b -- selection along NA sequence

    pprops = { 'xticks':      [],
               'xticklabels': [],
               'xlim':        [0, np.max(p[1])+1],
               'yticks':      [0, 0.2],
               'yminorticks': [0.05, 0.1, 0.15],
               'nudgey':      1.05,
               'xlabel':      'Protein position',
               'ylabel':      'Relative\nselection in NA',
               'plotprops':   lineprops,
               'axoffset':    0.1,
               'hide':        ['bottom'],
               'theme':       'open' }
    
    mp.line(             ax=ax_sna, x=[p[1]], y=[s[1][0]], colors=[colors[1][0]], **pprops)
    mp.plot(type='line', ax=ax_sna, x=[p[1]], y=[s[1][1]], colors=[colors[1][1]], **pprops)

    ax_sna.text(box_sna['left']+0.01, box_sna['top']-0.02, 'N2', color=colors[1][0], transform=fig.transFigure, **txtprops)
    ax_sna.text(box_sna['left']+0.01, box_sna['top']-0.07, 'N1', color=colors[1][1], transform=fig.transFigure, **txtprops)

    # analysis -- correlation between selection in N2 and N1 at same positions
    print('Correlation between selection in N2 and N1')
    print('domain\ttype\t\tr\tp')
    for domain in ['full']:
        if domain=='full':
            idxs = np.arange(len(p[1]))
        for typ in ['pearson', 'spearman']:
            if typ=='pearson':
                r, pval = st.pearsonr(s[1][0][idxs], s[1][1][idxs])
            elif typ=='spearman':
                r, pval = st.spearmanr(s[1][0][idxs], s[1][1][idxs])
            print('%s\t%s  \t%.3f\t%.2e' % (domain, typ, r, pval))

    # # c -- selection in HA over time

    # # make meshgrid with protein position on x axis and time on y axis
    # y, x = np.mgrid[0:np.max(t[0]), 0:np.max(p[0])+1]
    # dsdt_max = np.max([dsdt[0].max(), dsdt[1].max()])

    # c = ax_tha[0].imshow(dsdt[0], cmap='Blues', vmin=0, vmax=dsdt_max, extent=[x.min(), x.max(), y.min(), y.max()],
    #           interpolation='nearest', origin='lower', aspect='auto')

    # pprops = { 'xticks':      [],
    #            'xticklabels': [],
    #            'xlim':        [0, len(dsdt[0][0])],
    #            'yticks':      [np.min(t[0]), np.max(t[0])],
    #         #    'yminorticks': [0.05, 0.1, 0.15],
    #         #    'nudgey':      1.05,
    #            'xlabel':      'Protein position',
    #            'ylabel':      'Time (days since Jan 1, 2009)',
    #            'plotprops':   lineprops,
    #            'axoffset':    0.1,
    #         #    'hide':        ['bottom'],
    #            'theme':       'open' }
    
    # mp.plot(type='line', ax=ax_tha[0], x=[[-1, -1]], y=[[-1, -1]], colors=[BKCOLOR], **pprops)

    ax_sha.text(box_sha['left']+dx_label, box_sha['top']+dy_label, 'a'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_sna.text(box_sna['left']+dx_label, box_sna['top']+dy_label, 'b'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    
    # ax_sel[0].text(box_sel['left']-0.06, (box_sel['top']+box_sel['bottom'])/2, 'Frequency', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)
    # ax_sel[0].text(box_sel['left']+dx_label-0.02, box_sel['top']+dy_label, 'b'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_emd_vs_mlr(emd_files, t_min=0, deltats=None, null_col='MLR_remove_rare_mut', sim_col='FM_remove_rare_mut', colors=None, label_reassortment=True, fig_name='fig-emd-vs-mlr'):
    """ Curves showing earthmover's distance between true and predicted/MLR population distributions. """

    # Read in data
    df = [pd.read_csv(f, comment='#', memory_map=True) for f in emd_files]
    if t_min>0 and 'Obs_Time' in df[0]:
        df = [d[d['Obs_Time']>t_min] for d in df]

    labels = ['H3', 'N2', 'H1', 'N1']
    if not deltats:
        deltats = [0 for i in range(len(df))]
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = DOUBLE_COLUMN
    goldh = w/1.9
    fig   = plt.figure(figsize=(w, goldh))
    box_top = 0.95
    box_bot = 0.15
    box_lft = 0.09
    box_dx  = 0.08

    box_traj = dict(left=box_lft,                  right=0.65, top=box_top, bottom=box_bot)
    box_dist = dict(left=box_traj['right']+box_dx, right=0.95, top=box_top, bottom=box_bot)

    hspace = 0.40

    gs_traj  = gridspec.GridSpec(4, 1, hspace=hspace, **box_traj)
    ax_traj  = [plt.subplot(gs_traj[i, 0]) for i in range(4)]
    gs_dist  = gridspec.GridSpec(4, 1, hspace=hspace, **box_dist)
    ax_dist  = [plt.subplot(gs_dist[i, 0]) for i in range(4)]

    ## a -- EMD over time

    if not colors:
        colors = sns.husl_palette(len(df))

    lineprops = { 'lw': SIZELINE*2.5, 'linestyle': '-', 'alpha': 1.0 }
    dashprops = { 'lw': SIZELINE*1.0, 'linestyle': ':', 'alpha': 0.7 }
    thindashprops = { 'lw': SIZELINE*1.0, 'linestyle': ':', 'alpha': 0.2 }

    xticks      = np.linspace(np.min(df[0]['Obs_Time']), np.max(df[0]['Obs_Time'])+1, num=8)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    # xticklabels = [f'{i.year}-{i.month}' for i in xticklabels]
    # t_start = dt.datetime(2009, 1, 1)
    # xticks  = [i*2*365 + 3*365 for i in range(6)]
    # xticklabels = [t_start + dt.timedelta(days=int(i)) for i in xticks]
    xticklabels = [i.strftime('%b %Y') for i in xticklabels]

    for i in range(len(df)):
        pprops = { 'colors':      [colors[i]],
                   'xlim':        [np.min(xticks), np.max(xticks) + 50],
                   'xticks':      xticks,
                   'xticklabels': xticklabels,
                   'ylim':        [-5.7,  5.7],
                   'yticks':      [-4, -2, 0, 2, 4],
                   'yminorticks': [-5.7, -5, -3, -1, 1, 3, 5, 5.7],
                   'axoffset':    0,
                   'hide':        ['top', 'right'],
                   'theme':       'open' }

        xs = [df[i]['Obs_Time'] + deltats[i]]
        ys = [df[i][null_col]-df[i][sim_col]]

        # print(np.max(ys[0]), np.min(ys[0]))
        # print(np.min(df[i]['Obs_Time'] + deltats[i]), np.max(df[i]['Obs_Time'] + deltats[i]))

        if i==len(df)-1:
            pprops['xlabel'] = 'Time prediction made'
        else:
            pprops['xticklabels'] = ['' for k in xticklabels]

        mp.plot(type='line', ax=ax_traj[i], x=xs, y=ys, plotprops=lineprops, **pprops)

        pprops['colors'] = [LCOLOR]
        xmin, xmax = ax_traj[i].get_xlim()
        mp.line(ax=ax_traj[i], x=[[xmin, xmax]], y=[[0, 0]], plotprops=dashprops, **pprops)
        for yy in [-4, -2, 2, 4]:
            mp.line(ax=ax_traj[i], x=[[xmin, xmax]], y=[[yy, yy]], plotprops=thindashprops, **pprops)

    ax_traj[0].text(box_traj['left']-0.06, (box_traj['top']+box_traj['bottom'])/2, 'Model improvement vs. MLR\ndistance to future ' + r'$(\Delta EMD)$', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)

    ## label reassortment event
    if label_reassortment:
        x_scale = 2200
        x_offset = 1100

        label_x = (np.array([0.95, 0.96, 0.97])) * x_scale + x_offset
        label_y = np.array([-2.7, -3.2, -3.2])

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'color': BKCOLOR, 'ha': 'left', 'va': 'center', 'clip_on': False}
        ax_traj[0].text(label_x[-1]+10, label_y[-1], 'reassortment', **txtprops)
        mp.line(ax=ax_traj[0], x=[label_x], y=[label_y], colors=[BKCOLOR], plotprops=dict(lw=AXWIDTH, ls='-', clip_on=False))

    ## b -- EMD distributions

    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=1, edgecolor='none', orientation='vertical', align='mid')

    for i in range(len(df)):
        pprops = { 'xlim'        : [-5.65, 5.65],
                   'xticks'      : [-4, -2, 0, 2, 4],
                   'xminorticks' : [-5.65, -5, -3, -1, 1, 3, 5, 5.65],
                   'ylim'        : [0, 0.25],
                   'yticks'      : [0, 0.1, 0.20],
                   'yminorticks' : [0.05, 0.15, 0.25],
                   'bins'        : np.arange(-5.65, 5.65+0.25, 0.25),
                   'combine'     : True,
                   'plotprops'   : histprops,
                   'axoffset'    : 0.1,
                   'theme'       : 'open' }

        if i==len(df)-1:
            pprops['xlabel'] = 'Model improvement vs. MLR\ndistance to future ' + r'$(\Delta EMD)$'
        else:
            pprops['xticklabels'] = ['' for k in pprops['xticks']]
        
        mp.plot(type='hist', ax=ax_dist[i], x=[df[i][null_col]-df[i][sim_col]], colors=[colors[i]], **pprops)
        ax_dist[i].axvline(x=0, color=LCOLOR, **dashprops)

        xmin, xmax = ax_dist[i].get_xlim()
        if np.min(df[i][null_col]-df[i][sim_col])<xmin or np.max(df[i][null_col]-df[i][sim_col])>xmax:
            print('Warning: histogram for %s is clipping data outside xlim [%.1f, %.1f]' % (labels[i], xmin, xmax))

        t, p = st.ttest_1samp(df[i][null_col]-df[i][sim_col], 0)
        p_exp = np.floor(np.log10(p))

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'right', 'va': 'center', 'clip_on': False}
        ax_dist[i].text(box_dist['right']+0.005, box_dist['top']-(box_dist['top']-box_dist['bottom']+0.06)*(i/4)-0.02, '%s' % labels[i], color=colors[i], fontweight='bold', transform=fig.transFigure, **txtprops)
        ax_dist[i].text(box_dist['right']+0.005, box_dist['top']-(box_dist['top']-box_dist['bottom']+0.06)*(i/4)-0.06, r'$p < 10^{%d}$' % (p_exp+1), color=BKCOLOR, transform=fig.transFigure, **txtprops)
        

    ax_dist[0].text(box_dist['left']-0.05, (box_dist['top']+box_dist['bottom'])/2, 'Frequency', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)
    
    ax_traj[0].text(box_traj['left']-0.05, box_traj['top']+0.02, 'a'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_dist[0].text(box_dist['left']-0.05, box_dist['top']+0.02, 'b'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # Statistics

    for i in range(len(df)):
        print('--- %s ---' % labels[i])
        print('n = %d timepoints' % len(df[i]))
        print('Model mean EMD: %.2f' % np.mean(df[i][sim_col]))
        print('MLR   mean EMD: %.2f' % np.mean(df[i][null_col]))
        print('Model median EMD: %.2f' % np.median(df[i][sim_col]))
        print('MLR   median EMD: %.2f' % np.median(df[i][null_col]))
        t, p = st.ttest_rel(df[i][sim_col], df[i][null_col])
        print('Paired t-test: t = %.2f, p = %.2e' % (t, p))
        print('Mean improvement in EMD: %.2f' % np.mean(df[i][null_col]-df[i][sim_col]))
        print('Median improvement in EMD: %.2f' % np.median(df[i][null_col]-df[i][sim_col]))
        print('')

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_emd_vs_ss(emd_files, t_min=0, deltats=None, null_col='SLM', sim_col='FM', colors=None, label_reassortment=True, fig_name='fig-emd-vs-ss'):
    """ Curves showing earthmover's distance between true and predicted/single site population distributions. """

    # Read in data
    df = [pd.read_csv(f, comment='#', memory_map=True) for f in emd_files]
    if t_min>0 and 'Obs_Time' in df[0]:
        df = [d[d['Obs_Time']>t_min] for d in df]

    if 'Obs_Time' not in df[0].columns:
        # add Obs_Time column if not present by counting days since 01/01/2009 without using a lambda function
        # observation date is in obs_date column in format YYYY-MM-DD
        for i in range(len(df)):
            df[i]['Obs_Time'] = [df[i]['obs_date'].iloc[j] for j in range(len(df[i]))]
            df[i]['Obs_Time'] = pd.to_datetime(df[i]['Obs_Time'], format='%m/%d/%y')
            df[i]['Obs_Time'] = (df[i]['Obs_Time'] - dt.datetime(2009, 1, 1)).dt.days

    labels = ['H3', 'N2', 'H1', 'N1']
    if not deltats:
        deltats = [0 for i in range(len(df))]
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = DOUBLE_COLUMN
    goldh = w/1.9
    fig   = plt.figure(figsize=(w, goldh))
    box_top = 0.95
    box_bot = 0.15
    box_lft = 0.09
    box_dx  = 0.08

    box_traj = dict(left=box_lft,                  right=0.65, top=box_top, bottom=box_bot)
    box_dist = dict(left=box_traj['right']+box_dx, right=0.95, top=box_top, bottom=box_bot)

    hspace = 0.40

    gs_traj  = gridspec.GridSpec(4, 1, hspace=hspace, **box_traj)
    ax_traj  = [plt.subplot(gs_traj[i, 0]) for i in range(4)]
    gs_dist  = gridspec.GridSpec(4, 1, hspace=hspace, **box_dist)
    ax_dist  = [plt.subplot(gs_dist[i, 0]) for i in range(4)]

    ## a -- EMD over time

    if not colors:
        colors = sns.husl_palette(len(df))

    lineprops = { 'lw': SIZELINE*2.5, 'linestyle': '-', 'alpha': 1.0 }
    dashprops = { 'lw': SIZELINE*1.0, 'linestyle': ':', 'alpha': 0.7 }
    thindashprops = { 'lw': SIZELINE*1.0, 'linestyle': ':', 'alpha': 0.2 }

    xticks      = np.linspace(np.min(df[0]['Obs_Time']), np.max(df[0]['Obs_Time'])+1, num=8)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    # xticklabels = [f'{i.year}-{i.month}' for i in xticklabels]
    # t_start = dt.datetime(2009, 1, 1)
    # xticks  = [i*2*365 + 3*365 for i in range(6)]
    # xticklabels = [t_start + dt.timedelta(days=int(i)) for i in xticks]
    xticklabels = [i.strftime('%b %Y') for i in xticklabels]

    for i in range(len(df)):
        pprops = { 'colors':      [colors[i]],
                   'xlim':        [np.min(xticks), np.max(xticks) + 50],
                   'xticks':      xticks,
                   'xticklabels': xticklabels,
                   'ylim':        [-5.7,  5.7],
                   'yticks':      [-4, -2, 0, 2, 4],
                   'yminorticks': [-5.7, -5, -3, -1, 1, 3, 5, 5.7],
                   'axoffset':    0,
                   'hide':        ['top', 'right'],
                   'theme':       'open' }

        xs = [df[i]['Obs_Time'] + deltats[i]]
        ys = [df[i][null_col]-df[i][sim_col]]

        # print(np.max(ys[0]), np.min(ys[0]))
        # print(np.min(df[i]['Obs_Time'] + deltats[i]), np.max(df[i]['Obs_Time'] + deltats[i]))

        if i==len(df)-1:
            pprops['xlabel'] = 'Time prediction made'
        else:
            pprops['xticklabels'] = ['' for k in xticklabels]

        mp.plot(type='line', ax=ax_traj[i], x=xs, y=ys, plotprops=lineprops, **pprops)

        pprops['colors'] = [LCOLOR]
        xmin, xmax = ax_traj[i].get_xlim()
        mp.line(ax=ax_traj[i], x=[[xmin, xmax]], y=[[0, 0]], plotprops=dashprops, **pprops)
        for yy in [-4, -2, 2, 4]:
            mp.line(ax=ax_traj[i], x=[[xmin, xmax]], y=[[yy, yy]], plotprops=thindashprops, **pprops)

    ax_traj[0].text(box_traj['left']-0.06, (box_traj['top']+box_traj['bottom'])/2, 'Model improvement vs. single site\ndistance to future ' + r'$(\Delta EMD)$', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)

    ## label reassortment event
    if label_reassortment:
        x_scale = 2200
        x_offset = 1100

        label_x = (np.array([0.95, 0.96, 0.97])) * x_scale + x_offset
        label_y = np.array([-2.7, -3.2, -3.2])

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'color': BKCOLOR, 'ha': 'left', 'va': 'center', 'clip_on': False}
        ax_traj[0].text(label_x[-1]+10, label_y[-1], 'reassortment', **txtprops)
        mp.line(ax=ax_traj[0], x=[label_x], y=[label_y], colors=[BKCOLOR], plotprops=dict(lw=AXWIDTH, ls='-', clip_on=False))

    ## b -- EMD distributions

    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=1, edgecolor='none', orientation='vertical', align='mid')

    for i in range(len(df)):
        pprops = { 'xlim'        : [-5.65, 5.65],
                   'xticks'      : [-4, -2, 0, 2, 4],
                   'xminorticks' : [-5.65, -5, -3, -1, 1, 3, 5, 5.65],
                   'ylim'        : [0, 0.25],
                   'yticks'      : [0, 0.1, 0.20],
                   'yminorticks' : [0.05, 0.15, 0.25],
                   'bins'        : np.arange(-5.65, 5.65+0.25, 0.25),
                   'combine'     : True,
                   'plotprops'   : histprops,
                   'axoffset'    : 0.1,
                   'theme'       : 'open' }

        if i==len(df)-1:
            pprops['xlabel'] = 'Model improvement vs. single site\ndistance to future ' + r'$(\Delta EMD)$'
        else:
            pprops['xticklabels'] = ['' for k in pprops['xticks']]
        
        mp.plot(type='hist', ax=ax_dist[i], x=[df[i][null_col]-df[i][sim_col]], colors=[colors[i]], **pprops)
        ax_dist[i].axvline(x=0, color=LCOLOR, **dashprops)

        xmin, xmax = ax_dist[i].get_xlim()
        if np.min(df[i][null_col]-df[i][sim_col])<xmin or np.max(df[i][null_col]-df[i][sim_col])>xmax:
            print('Warning: histogram for %s is clipping data outside xlim [%.1f, %.1f]' % (labels[i], xmin, xmax))

        t, p = st.ttest_1samp(df[i][null_col]-df[i][sim_col], 0)
        p_exp = np.floor(np.log10(p))

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'right', 'va': 'center', 'clip_on': False}
        ax_dist[i].text(box_dist['right']+0.005, box_dist['top']-(box_dist['top']-box_dist['bottom']+0.06)*(i/4)-0.02, '%s' % labels[i], color=colors[i], fontweight='bold', transform=fig.transFigure, **txtprops)
        ax_dist[i].text(box_dist['right']+0.005, box_dist['top']-(box_dist['top']-box_dist['bottom']+0.06)*(i/4)-0.06, r'$p < 10^{%d}$' % (p_exp+1), color=BKCOLOR, transform=fig.transFigure, **txtprops)
        

    ax_dist[0].text(box_dist['left']-0.05, (box_dist['top']+box_dist['bottom'])/2, 'Frequency', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)
    
    ax_traj[0].text(box_traj['left']-0.05, box_traj['top']+0.02, 'a'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_dist[0].text(box_dist['left']-0.05, box_dist['top']+0.02, 'b'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # Statistics

    for i in range(len(df)):
        print('--- %s ---' % labels[i])
        print('n = %d timepoints' % len(df[i]))
        print('Model mean EMD: %.2f' % np.mean(df[i][sim_col]))
        print('SS    mean EMD: %.2f' % np.mean(df[i][null_col]))
        print('Model median EMD: %.2f' % np.median(df[i][sim_col]))
        print('SS    median EMD: %.2f' % np.median(df[i][null_col]))
        t, p = st.ttest_rel(df[i][sim_col], df[i][null_col])
        print('Paired t-test: t = %.2f, p = %.2e' % (t, p))
        print('Mean improvement in EMD: %.2f' % np.mean(df[i][null_col]-df[i][sim_col]))
        print('Median improvement in EMD: %.2f' % np.median(df[i][null_col]-df[i][sim_col]))
        print('')

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_emd(emd_files, t_min=0, extend_limits=False, deltats=None, null_col='Null_Dist', sim_col='Medoid_Dist', colors=None, label_reassortment=True, fig_name='fig-emd'):
    """ Curves showing earthmover's distance between true and predicted/null population distributions. """

    # Read in data
    df = [pd.read_csv(f, comment='#', memory_map=True) for f in emd_files]
    if t_min>0 and 'Obs_Time' in df[0]:
        df = [d[d['Obs_Time']>t_min] for d in df]

    labels = ['H3', 'N2', 'H1', 'N1']
    if not deltats:
        deltats = [0 for i in range(len(df))]

    max_delta = 4.25
    if extend_limits:
        max_delta = 5.0
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = DOUBLE_COLUMN
    goldh = w/2.2
    fig   = plt.figure(figsize=(w, goldh))
    box_top = 0.95
    box_bot = 0.15
    box_lft = 0.09
    box_dx  = 0.08

    box_traj = dict(left=box_lft,                  right=0.65, top=box_top, bottom=box_bot)
    box_dist = dict(left=box_traj['right']+box_dx, right=0.95, top=box_top, bottom=box_bot)

    hspace = 0.40

    gs_traj  = gridspec.GridSpec(4, 1, hspace=hspace, **box_traj)
    ax_traj  = [plt.subplot(gs_traj[i, 0]) for i in range(4)]
    gs_dist  = gridspec.GridSpec(4, 1, hspace=hspace, **box_dist)
    ax_dist  = [plt.subplot(gs_dist[i, 0]) for i in range(4)]

    ## a -- EMD over time

    if not colors:
        colors = sns.husl_palette(len(df))

    lineprops = { 'lw': SIZELINE*2.5, 'linestyle': '-', 'alpha': 1.0 }
    dashprops = { 'lw': SIZELINE*1.0, 'linestyle': ':', 'alpha': 0.7 }
    thindashprops = { 'lw': SIZELINE*1.0, 'linestyle': ':', 'alpha': 0.2 }

    xticks      = np.linspace(np.min(df[0]['Obs_Time']), np.max(df[0]['Obs_Time'])+1, num=8)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    # xticklabels = [f'{i.year}-{i.month}' for i in xticklabels]
    # t_start = dt.datetime(2009, 1, 1)
    # xticks  = [i*2*365 + 3*365 for i in range(6)]
    # xticklabels = [t_start + dt.timedelta(days=int(i)) for i in xticks]
    xticklabels = [i.strftime('%b %Y') for i in xticklabels]

    for i in range(len(df)):
        pprops = { 'colors':      [colors[i]],
                   'xlim':        [np.min(xticks), np.max(xticks) + 50],
                   'xticks':      xticks,
                   'xticklabels': xticklabels,
                   'ylim':        [-max_delta, max_delta],
                   'yticks':      [-4, -2, 0, 2, 4],
                   'yminorticks': [-3, -1, 1, 3],
                   'axoffset':    0,
                   'hide':        ['top', 'right'],
                   'theme':       'open' }
        
        if extend_limits:
            pprops['yminorticks'] = [-5, -3, -1, 1, 3, 5]

        xs = [df[i]['Obs_Time'] + deltats[i]]
        ys = [df[i][null_col]-df[i][sim_col]]

        # print(np.max(ys[0]), np.min(ys[0]))
        # print(np.min(df[i]['Obs_Time'] + deltats[i]), np.max(df[i]['Obs_Time'] + deltats[i]))

        if i==len(df)-1:
            pprops['xlabel'] = 'Time prediction made'
        else:
            pprops['xticklabels'] = ['' for k in xticklabels]

        mp.plot(type='line', ax=ax_traj[i], x=xs, y=ys, plotprops=lineprops, **pprops)

        pprops['colors'] = [LCOLOR]
        xmin, xmax = ax_traj[i].get_xlim()
        mp.line(ax=ax_traj[i], x=[[xmin, xmax]], y=[[0, 0]], plotprops=dashprops, **pprops)
        for yy in [-4, -2, 2, 4]:
            mp.line(ax=ax_traj[i], x=[[xmin, xmax]], y=[[yy, yy]], plotprops=thindashprops, **pprops)

    ax_traj[0].text(box_traj['left']-0.06, (box_traj['top']+box_traj['bottom'])/2, 'Model improvement vs. neutral\ndistance to future ' + r'$(\Delta EMD)$', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)

    ## label reassortment event
    if label_reassortment:
        x_scale = 2200
        x_offset = 1100

        label_x = (np.array([0.95, 0.96, 0.97])) * x_scale + x_offset
        label_y = np.array([-2.7, -3.2, -3.2])

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'color': BKCOLOR, 'ha': 'left', 'va': 'center', 'clip_on': False}
        ax_traj[0].text(label_x[-1]+10, label_y[-1], 'reassortment', **txtprops)
        mp.line(ax=ax_traj[0], x=[label_x], y=[label_y], colors=[BKCOLOR], plotprops=dict(lw=AXWIDTH, ls='-', clip_on=False))

    ## b -- EMD distributions

    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=1, edgecolor='none', orientation='vertical', align='mid')

    for i in range(len(df)):
        pprops = { 'xlim'        : [-max_delta-0.25, max_delta+0.25],
                   'xticks'      : [-4, -2, 0, 2, 4],
                   'xminorticks' : [-3, -1, 1, 3],
                   'ylim'        : [0, 0.25],
                   'yticks'      : [0, 0.1, 0.20],
                   'yminorticks' : [0.05, 0.15, 0.25],
                   'bins'        : np.arange(-max_delta-0.25, max_delta+0.5, 0.25),
                   'combine'     : True,
                   'plotprops'   : histprops,
                   'axoffset'    : 0.1,
                   'theme'       : 'open' }
        
        if extend_limits:
            pprops['xminorticks'] = [-5, -3, -1, 1, 3, 5]

        if i==len(df)-1:
            pprops['xlabel'] = 'Model improvement vs. neutral\ndistance to future ' + r'$(\Delta EMD)$'
        else:
            pprops['xticklabels'] = ['' for k in pprops['xticks']]
        
        mp.plot(type='hist', ax=ax_dist[i], x=[df[i][null_col]-df[i][sim_col]], colors=[colors[i]], **pprops)
        ax_dist[i].axvline(x=0, color=LCOLOR, **dashprops)

        xmin, xmax = ax_dist[i].get_xlim()
        if np.min(df[i][null_col]-df[i][sim_col])<xmin or np.max(df[i][null_col]-df[i][sim_col])>xmax:
            print('Warning: histogram for %s is clipping data outside xlim [%.1f, %.1f]' % (labels[i], xmin, xmax))

        t, p = st.ttest_1samp(df[i][null_col]-df[i][sim_col], 0)
        p_exp = np.floor(np.log10(p))

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'right', 'va': 'center', 'clip_on': False}
        ax_dist[i].text(box_dist['right']+0.005, box_dist['top']-(box_dist['top']-box_dist['bottom']+0.06)*(i/4)-0.01, '%s' % labels[i], color=colors[i], fontweight='bold', transform=fig.transFigure, **txtprops)
        ax_dist[i].text(box_dist['right']+0.005, box_dist['top']-(box_dist['top']-box_dist['bottom']+0.06)*(i/4)-0.05, r'$p < 10^{%d}$' % (p_exp+1), color=BKCOLOR, transform=fig.transFigure, **txtprops)
        

    ax_dist[0].text(box_dist['left']-0.05, (box_dist['top']+box_dist['bottom'])/2, 'Frequency', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)
    
    ax_traj[0].text(box_traj['left']-0.05, box_traj['top']+0.02, 'a'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_dist[0].text(box_dist['left']-0.05, box_dist['top']+0.02, 'b'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # Statistics

    for i in range(len(df)):
        print('--- %s ---' % labels[i])
        print('n = %d timepoints' % len(df[i]))
        print('Model mean EMD: %.2f' % np.mean(df[i][sim_col]))
        print('Null  mean EMD: %.2f' % np.mean(df[i][null_col]))
        print('Model median EMD: %.2f' % np.median(df[i][sim_col]))
        print('Null  median EMD: %.2f' % np.median(df[i][null_col]))
        t, p = st.ttest_rel(df[i][sim_col], df[i][null_col])
        print('Paired t-test: t = %.2f, p = %.2e' % (t, p))
        print('Mean improvement in EMD: %.2f' % np.mean(df[i][null_col]-df[i][sim_col]))
        print('Median improvement in EMD: %.2f' % np.median(df[i][null_col]-df[i][sim_col]))
        print('')

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_emd_clade(emd_files, df_past, df_full, df_forecast, t_min=0, deltats=None, null_col='Null_Dist', sim_col='Medoid_Dist', colors=None, label_reassortment=True, fig_name='fig-emd'):
    """ Curves showing earthmover's distance between true and predicted/null population distributions. """

    # Read in data
    df = [pd.read_csv(f, comment='#', memory_map=True) for f in emd_files]
    if t_min>0 and 'Obs_Time' in df[0]:
        df = [d[d['Obs_Time']>t_min] for d in df]

    labels = ['H3', 'N2', 'H1', 'N1']
    if not deltats:
        deltats = [0 for i in range(len(df))]
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = DOUBLE_COLUMN
    goldh = w/1.6
    fig   = plt.figure(figsize=(w, goldh))
    box_top = 0.95
    box_bot = 0.95 - 0.8 * (1.6/2.2)
    box_lft = 0.09
    box_dx  = 0.08

    box_traj = dict(left=box_lft,                  right=0.65, top=box_top, bottom=box_bot)
    box_dist = dict(left=box_traj['right']+box_dx, right=0.95, top=box_top, bottom=box_bot)

    hspace = 0.40

    gs_traj  = gridspec.GridSpec(4, 1, hspace=hspace, **box_traj)
    ax_traj  = [plt.subplot(gs_traj[i, 0]) for i in range(4)]
    gs_dist  = gridspec.GridSpec(4, 1, hspace=hspace, **box_dist)
    ax_dist  = [plt.subplot(gs_dist[i, 0]) for i in range(4)]

    ## a -- EMD over time

    if not colors:
        colors = sns.husl_palette(len(df))

    lineprops = { 'lw': SIZELINE*2.5, 'linestyle': '-', 'alpha': 1.0 }
    dashprops = { 'lw': SIZELINE*1.0, 'linestyle': ':', 'alpha': 0.5 }
    thindashprops = { 'lw': SIZELINE*1.0, 'linestyle': ':', 'alpha': 0.2 }

    xticks      = np.linspace(np.min(df[0]['Obs_Time']), np.max(df[0]['Obs_Time'])+1, num=8)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    # xticklabels = [f'{i.year}-{i.month}' for i in xticklabels]
    # t_start = dt.datetime(2009, 1, 1)
    # xticks  = [i*2*365 + 3*365 for i in range(6)]
    # xticklabels = [t_start + dt.timedelta(days=int(i)) for i in xticks]
    xticklabels = [i.strftime('%b %Y') for i in xticklabels]

    for i in range(len(df)):
        pprops = { 'colors':      [colors[i]],
                   'xlim':        [np.min(xticks), np.max(xticks) + 50],
                   'xticks':      xticks,
                   'xticklabels': xticklabels,
                   'ylim':        [-4.2,  4.2],
                   'yticks':      [-4, -2, 0, 2, 4],
                   'yminorticks': [-3, -1, 1, 3],
                   'axoffset':    0,
                   'hide':        ['top', 'right'],
                   'theme':       'open' }

        xs = [df[i]['Obs_Time'] + deltats[i]]
        ys = [df[i][null_col]-df[i][sim_col]]

        # print(np.min(df[i]['Obs_Time'] + deltats[i]), np.max(df[i]['Obs_Time'] + deltats[i]))

        if i==len(df)-1:
            pprops['xlabel'] = 'Time prediction made'
        else:
            pprops['xticklabels'] = ['' for k in xticklabels]

        mp.plot(type='line', ax=ax_traj[i], x=xs, y=ys, plotprops=lineprops, **pprops)

        pprops['colors'] = [LCOLOR]
        xmin, xmax = ax_traj[i].get_xlim()
        mp.line(ax=ax_traj[i], x=[[xmin, xmax]], y=[[0, 0]], plotprops=dashprops, **pprops)
        for yy in [-4, -2, 2, 4]:
            mp.line(ax=ax_traj[i], x=[[xmin, xmax]], y=[[yy, yy]], plotprops=thindashprops, **pprops)

    ax_traj[0].text(box_traj['left']-0.06, (box_traj['top']+box_traj['bottom'])/2, 'Model improvement over null\ndistance to future ' + r'$(\Delta EMD)$', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)

    ## label reassortment event
    if label_reassortment:
        x_scale = 2200
        x_offset = 1100

        label_x = (np.array([0.95, 0.96, 0.97])) * x_scale + x_offset
        label_y = np.array([-2.7, -3.2, -3.2])

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'color': BKCOLOR, 'ha': 'left', 'va': 'center', 'clip_on': False}
        ax_traj[0].text(label_x[-1]+10, label_y[-1], 'reassortment', **txtprops)
        mp.line(ax=ax_traj[0], x=[label_x], y=[label_y], colors=[BKCOLOR], plotprops=dict(lw=AXWIDTH, ls='-', clip_on=False))

    ## b -- EMD distributions

    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=1, edgecolor='none', orientation='vertical')

    for i in range(len(df)):
        pprops = { 'xticks'      : [-4, -2, 0, 2, 4],
                #    'ylim'        : [0.01, 1],
                #    'yticks'      : [0.01, 0.1, 1],
                   'ylim'        : [0, 0.25],
                   'yticks'      : [0, 0.1, 0.20],
                   'yminorticks' : [0.05, 0.15, 0.25],
                   'bins'        : np.arange(-4, 4+0.25, 0.25),
                   'combine'     : True,
                   'plotprops'   : histprops,
                   'axoffset'    : 0.1,
                   'theme'       : 'open' }

        # ax_dist[i].set_yscale('log')
        if i==len(df)-1:
            pprops['xlabel'] = 'Model improvement over null\ndistance to future ' + r'$(\Delta EMD)$'
        else:
            pprops['xticklabels'] = ['' for k in pprops['xticks']]
        
        mp.plot(type='hist', ax=ax_dist[i], x=[df[i][null_col]-df[i][sim_col]], colors=[colors[i]], **pprops)
        ax_dist[i].axvline(x=0, color=LCOLOR, **dashprops)

        xmin, xmax = ax_dist[i].get_xlim()
        if np.min(df[i][null_col]-df[i][sim_col])<xmin or np.max(df[i][null_col]-df[i][sim_col])>xmax:
            print('Warning: histogram for %s is clipping data outside xlim [%.1f, %.1f]' % (labels[i], xmin, xmax))

        t, p = st.ttest_1samp(df[i][null_col]-df[i][sim_col], 0)
        p_exp = np.floor(np.log10(p))

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'right', 'va': 'center', 'clip_on': False}
        ax_dist[i].text(box_dist['right']+0.005, box_dist['top']-(box_dist['top']-box_dist['bottom']+0.06)*(i/4), '%s' % labels[i], color=colors[i], fontweight='bold', transform=fig.transFigure, **txtprops)
        ax_dist[i].text(box_dist['right']+0.005, box_dist['top']-(box_dist['top']-box_dist['bottom']+0.06)*(i/4)-0.04, r'$p < 10^{%d}$' % (p_exp+1), color=BKCOLOR, transform=fig.transFigure, **txtprops)

    ax_dist[0].text(box_dist['left']-0.05, (box_dist['top']+box_dist['bottom'])/2, 'Frequency', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)
    
    ax_traj[0].text(box_traj['left']-0.05, box_traj['top']+0.02, 'a'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_dist[0].text(box_dist['left']-0.05, box_dist['top']+0.02, 'b'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # Statistics

    for i in range(len(df)):
        print('--- %s ---' % labels[i])
        print('n = %d timepoints' % len(df[i]))
        print('Model mean EMD: %.2f' % np.mean(df[i][sim_col]))
        print('Null  mean EMD: %.2f' % np.mean(df[i][null_col]))
        print('Model median EMD: %.2f' % np.median(df[i][sim_col]))
        print('Null  median EMD: %.2f' % np.median(df[i][null_col]))
        t, p = st.ttest_rel(df[i][sim_col], df[i][null_col])
        print('Paired t-test: t = %.2f, p = %.2e' % (t, p))
        print('Mean improvement in EMD: %.2f' % np.mean(df[i][null_col]-df[i][sim_col]))
        print('Median improvement in EMD: %.2f' % np.median(df[i][null_col]-df[i][sim_col]))
        print('')

    ## c -- clade frequency

    for df in [df_past, df_full] + df_forecast:
        df['Date'] = pd.to_datetime(df['Date'])
        # convert date to days since 2009-01-01
        df['Date'] = (df['Date'] - dt.datetime(2009, 1, 1)).dt.days

    df_fc = pd.concat(df_forecast)
    
    for df in [df_past, df_full, df_fc]:
        df['clade'] = df['clade'].astype(str).str.strip()
        df['frequency'] = df.groupby('Date')['frequency'].transform(lambda x: x / x.sum())

    pivot_past = df_past.pivot_table(index='Date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot_past = pivot_past.sort_index()
    pivot_full = df_full.pivot_table(index='Date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot_full = pivot_full.sort_index()
    pivot_fc = df_fc.pivot_table(index='Date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot_fc = pivot_fc.sort_index()

    label_threshold = 0.01
    max_freqs = pivot_past.max()
    clades_to_plot = max_freqs[max_freqs > label_threshold].index
    pivot_past = pivot_past[clades_to_plot]
    pivot_full = pivot_full[clades_to_plot]
    pivot_fc = pivot_fc[clades_to_plot]

    x = pivot_past.index
    Y = pivot_past.values.T
    clades = pivot_past.columns

    past_len = df_past['Date'].max() - df_past['Date'].min()
    fc_min = df_past['Date'].max()
    fc_max = df_fc['Date'].max()
    fc_len = fc_max - fc_min

    total_len = past_len + (3 * fc_len)
    box_dx2 = 0.05
    total_dx = 0.85 - box_lft - (3 * box_dx2)
    past_dx = total_dx * (past_len / total_len)
    fc_dx = (total_dx - past_dx) / 3
    box_dy = 0.15 
    box_b2 = 0.05

    box_clade_past = dict(left=box_lft, right=box_lft + past_dx, top=box_bot-box_dy, bottom=box_b2)
    box_clade_fc   = dict(left=box_clade_past['right']+box_dx2, right=box_clade_past['right']+box_dx2+fc_dx, top=box_bot-box_dy, bottom=box_b2)
    box_clade_true = dict(left=box_clade_fc['right']+box_dx2, right=box_clade_fc['right']+box_dx2+fc_dx, top=box_bot-box_dy, bottom=box_b2)
    box_clade_null = dict(left=box_clade_true['right']+box_dx2, right=box_clade_true['right']+box_dx2+fc_dx, top=box_bot-box_dy, bottom=box_b2)

    gs_clade_past = gridspec.GridSpec(1, 1, **box_clade_past)
    ax_clade_past = plt.subplot(gs_clade_past[0, 0])
    gs_clade_fc   = gridspec.GridSpec(1, 1, **box_clade_fc)
    ax_clade_fc   = plt.subplot(gs_clade_fc[0, 0])
    gs_clade_true = gridspec.GridSpec(1, 1, **box_clade_true)
    ax_clade_true = plt.subplot(gs_clade_true[0, 0])
    gs_clade_null = gridspec.GridSpec(1, 1, **box_clade_null)
    ax_clade_null = plt.subplot(gs_clade_null[0, 0])

    # print(df_past.head())
    # print(pivot_past.head())

    ## past clade frequencies

    colors = sns.husl_palette(len(clades))

    xticks      = np.linspace(np.min(df_past['Date']), np.max(df_past['Date'])+1, num=4)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    # xticklabels = [f'{i.year}-{i.month}' for i in xticklabels]
    # t_start = dt.datetime(2009, 1, 1)
    # xticks  = [i*2*365 + 3*365 for i in range(6)]
    # xticklabels = [t_start + dt.timedelta(days=int(i)) for i in xticks]
    xticklabels = [i.strftime('%b %Y') for i in xticklabels]

    pprops = { 'xlim':        [np.min(xticks), np.max(xticks)],
                'xticks':      xticks,
                'xticklabels': xticklabels,
                'ylim':        [0, 1],
                'yticks':      [0, 0.5, 1],
                'yminorticks': [0.25, 0.75],
                'ylabel':      'Clade frequency',
                'axoffset':    0,
                # 'hide':        ['top', 'right'],
                'theme':       'boxed' }

    ax_clade_past.stackplot(x, Y, labels=clades, colors=colors, edgecolors='w', lw=AXWIDTH)
    mp.plot(type='line', ax=ax_clade_past, x=[[-1, -1]], y=[[-1, -1]], plotprops=DEF_LINEPROPS, **pprops)
    ax_clade_past.tick_params(axis='x', pad=8)

    ## forecast clade frequencies

    df_last = df_past[df_past['Date'] == df_past['Date'].max()]
    df_fc = pd.concat([df_last, df_fc])

    pivot_fc = df_fc.pivot_table(index='Date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot_fc = pivot_fc.sort_index()
    pivot_fc = pivot_fc[clades_to_plot]
    x = pivot_fc.index
    Y = pivot_fc.values.T

    xticks      = np.linspace(np.min(df_fc['Date']), np.max(df_fc['Date'])+1, num=5)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    # xticklabels = [xticklabels[i].strftime('%b %Y') if i in (0, len(xticklabels)-1) else '' for i in range(len(xticklabels))]
    xticklabels = ['', '3m', '6m', '9m', '12m']

    pprops = { 'xlim':        [np.min(xticks), np.max(xticks)],
                'xticks':      xticks,
                'xticklabels': xticklabels,
                'ylim':        [0, 1],
                'yticks':      [0, 0.5, 1],
                'yticklabels': ['', '', ''],
                'yminorticks': [0.25, 0.75],
                # 'ylabel':      'Clade frequency',
                'axoffset':    0,
                # 'hide':        ['top', 'right'],
                'theme':       'boxed' }

    ax_clade_fc.stackplot(x, Y, labels=clades, colors=colors, edgecolors='w', lw=AXWIDTH)
    mp.plot(type='line', ax=ax_clade_fc, x=[[-1, -1]], y=[[-1, -1]], plotprops=DEF_LINEPROPS, **pprops)
    ax_clade_fc.tick_params(axis='x', pad=8)

    ## true clade frequencies

    pivot_true = df_full.pivot_table(index='Date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot_true = pivot_true.sort_index()
    pivot_true = pivot_true[clades_to_plot]
    x = pivot_true.index
    Y = pivot_true.values.T

    xticks      = np.linspace(np.min(df_fc['Date']), np.max(df_fc['Date'])+1, num=5)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    # xticklabels = [xticklabels[i].strftime('%b %Y') if i in (0, len(xticklabels)-1) else '' for i in range(len(xticklabels))]
    xticklabels = ['', '3m', '6m', '9m', '12m']

    pprops = { 'xlim':        [np.min(xticks), np.max(xticks)],
                'xticks':      xticks,
                'xticklabels': xticklabels,
                'ylim':        [0, 1],
                'yticks':      [0, 0.5, 1],
                'yticklabels': ['', '', ''],
                'yminorticks': [0.25, 0.75],
                # 'ylabel':      'Clade frequency',
                'axoffset':    0,
                # 'hide':        ['top', 'right'],
                'theme':       'boxed' }

    ax_clade_true.stackplot(x, Y, labels=clades, colors=colors, edgecolors='w', lw=AXWIDTH)
    mp.plot(type='line', ax=ax_clade_true, x=[[-1, -1]], y=[[-1, -1]], plotprops=DEF_LINEPROPS, **pprops)
    ax_clade_true.tick_params(axis='x', pad=8)

    ## null clade frequencies -- same as last observed in df_past

    x = df_fc['Date'].unique()
    x.sort()
    Y = np.array([df_last[df_last['clade'] == clade]['frequency'].values[0] for clade in clades_to_plot])
    Y = np.tile(Y, (len(x), 1)).T

    xticks      = np.linspace(np.min(df_fc['Date']), np.max(df_fc['Date'])+1, num=5)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    # xticklabels = [xticklabels[i].strftime('%b %Y') if i in (0, len(xticklabels)-1) else '' for i in range(len(xticklabels))]
    xticklabels = ['', '3m', '6m', '9m', '12m']

    pprops = { 'xlim':        [np.min(xticks), np.max(xticks)],
                'xticks':      xticks,
                'xticklabels': xticklabels,
                'ylim':        [0, 1],
                'yticks':      [0, 0.5, 1],
                'yticklabels': ['', '', ''],
                'yminorticks': [0.25, 0.75],
                # 'ylabel':      'Clade frequency',
                'axoffset':    0,
                # 'hide':        ['top', 'right'],
                'theme':       'boxed' }

    ax_clade_null.stackplot(x, Y, labels=clades, colors=colors, edgecolors='w', lw=AXWIDTH)
    mp.plot(type='line', ax=ax_clade_null, x=[[-1, -1]], y=[[-1, -1]], plotprops=DEF_LINEPROPS, **pprops)
    ax_clade_null.tick_params(axis='x', pad=8)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_clade_old(df_past, df_full, df_forecast, fig_name='fig-clade'):
    """ Plot past, future, and predicted clade frequency example. """

    # Read in data
    for df in [df_past, df_full] + df_forecast:
        df['date'] = pd.to_datetime(df['date'])
        # convert date to days since 2009-01-01
        df['date'] = (df['date'] - dt.datetime(2009, 1, 1)).dt.days

    df_fc = pd.concat(df_forecast)
    
    for df in [df_past, df_full, df_fc]:
        df['clade'] = df['clade'].astype(str).str.strip()
        df['frequency'] = df.groupby('date')['frequency'].transform(lambda x: x / x.sum())

    pivot_past = df_past.pivot_table(index='date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot_past = pivot_past.sort_index()
    pivot_full = df_full.pivot_table(index='date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot_full = pivot_full.sort_index()
    pivot_fc = df_fc.pivot_table(index='date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot_fc = pivot_fc.sort_index()

    label_threshold = 0.01
    max_freqs = pivot_past.max()
    clades_to_plot = max_freqs[max_freqs > label_threshold].index
    pivot_past = pivot_past[clades_to_plot]
    pivot_full = pivot_full[clades_to_plot]
    pivot_fc = pivot_fc[clades_to_plot]

    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = ONE_FIVE_COLUMN
    goldh = w/1.6
    fig   = plt.figure(figsize=(w, goldh))
    box_top = 0.95
    box_bot = 0.08
    box_lft = 0.10
    box_rt  = 0.80
    box_dx  = 0.05
    box_dy  = 0.20

    past_len = df_past['date'].max() - df_past['date'].min()
    fc_min = df_past['date'].max()
    fc_max = df_fc['date'].max()
    fc_len = fc_max - fc_min

    total_dx_past = box_rt - box_lft
    total_dx_fc = 3 * (fc_len / past_len) * total_dx_past + 2 * box_dx
    if total_dx_fc > 1.0:
        print('Need to resize past box or reduce spacing to fit forecast boxes: total_dx_fc = %.2f' % total_dx_fc)

    fc_dx = total_dx_past * (fc_len / past_len)

    box_clade_past = dict(left=box_lft,                          right=box_rt,                                   top=box_top,                                bottom=box_top-(box_top - box_bot - box_dy)/2)
    box_clade_fc   = dict(left=box_lft,                          right=box_lft + fc_dx,                          top=box_top-(box_top - box_bot + box_dy)/2, bottom=box_bot)
    box_clade_true = dict(left=box_clade_fc['right'] + box_dx,   right=box_clade_fc['right'] + box_dx + fc_dx,   top=box_top-(box_top - box_bot + box_dy)/2, bottom=box_bot)
    box_clade_null = dict(left=box_clade_true['right'] + box_dx, right=box_clade_true['right'] + box_dx + fc_dx, top=box_top-(box_top - box_bot + box_dy)/2, bottom=box_bot)

    gs_clade_past = gridspec.GridSpec(1, 1, **box_clade_past)
    ax_clade_past = plt.subplot(gs_clade_past[0, 0])
    gs_clade_fc   = gridspec.GridSpec(1, 1, **box_clade_fc)
    ax_clade_fc   = plt.subplot(gs_clade_fc[0, 0])
    gs_clade_true = gridspec.GridSpec(1, 1, **box_clade_true)
    ax_clade_true = plt.subplot(gs_clade_true[0, 0])
    gs_clade_null = gridspec.GridSpec(1, 1, **box_clade_null)
    ax_clade_null = plt.subplot(gs_clade_null[0, 0])
    
    ## past clade frequencies

    x = pivot_past.index
    Y = pivot_past.values.T
    clades = pivot_past.columns

    # colors = sns.husl_palette(len(clades))
    # colors = sns.color_palette('crest', n_colors=len(clades))
    # colors = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(clades))
    colors = sns.cubehelix_palette(n_colors=len(clades))

    xticks      = np.linspace(np.min(df_past['Date']), np.max(df_past['Date'])+1, num=4)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    xticklabels = [i.strftime('%b %Y') for i in xticklabels]

    pprops = { 'xlim':        [np.min(xticks), np.max(xticks)],
                'xticks':      xticks,
                'xticklabels': xticklabels,
                'ylim':        [0, 1],
                'yticks':      [0, 0.5, 1],
                'yminorticks': [0.25, 0.75],
                'ylabel':      'Clade frequency',
                'axoffset':    0,
                'theme':       'boxed' }

    ax_clade_past.stackplot(x, Y, labels=clades, colors=colors, edgecolors='w', lw=AXWIDTH)
    mp.plot(type='line', ax=ax_clade_past, x=[[-1, -1]], y=[[-1, -1]], plotprops=DEF_LINEPROPS, **pprops)
    ax_clade_past.tick_params(axis='x', pad=8)

    ## forecast clade frequencies

    df_last = df_past[df_past['date'] == df_past['date'].max()]
    df_fc = pd.concat([df_last, df_fc])

    pivot_fc = df_fc.pivot_table(index='date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot_fc = pivot_fc.sort_index()
    pivot_fc = pivot_fc[clades_to_plot]
    x = pivot_fc.index
    Y = pivot_fc.values.T

    xticks      = np.linspace(np.min(df_fc['date']), np.max(df_fc['date'])+1, num=5)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    # xticklabels = [xticklabels[i].strftime('%b %Y') if i in (0, len(xticklabels)-1) else '' for i in range(len(xticklabels))]
    xticklabels = ['', '3m', '6m', '9m', '12m']

    pprops = { 'xlim':        [np.min(xticks), np.max(xticks)],
                'xticks':      xticks,
                'xticklabels': xticklabels,
                'ylim':        [0, 1],
                'yticks':      [0, 0.5, 1],
                # 'yticklabels': ['', '', ''],
                'yminorticks': [0.25, 0.75],
                'ylabel':      'Clade frequency',
                'axoffset':    0,
                'theme':       'boxed' }

    ax_clade_fc.stackplot(x, Y, labels=clades, colors=colors, edgecolors='w', lw=AXWIDTH)
    mp.plot(type='line', ax=ax_clade_fc, x=[[-1, -1]], y=[[-1, -1]], plotprops=DEF_LINEPROPS, **pprops)
    ax_clade_fc.tick_params(axis='x', pad=4)

    ## true clade frequencies

    pivot_true = df_full.pivot_table(index='date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot_true = pivot_true.sort_index()
    pivot_true = pivot_true[clades_to_plot]
    x = pivot_true.index
    Y = pivot_true.values.T

    xticks      = np.linspace(np.min(df_fc['date']), np.max(df_fc['date'])+1, num=5)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    # xticklabels = [xticklabels[i].strftime('%b %Y') if i in (0, len(xticklabels)-1) else '' for i in range(len(xticklabels))]
    xticklabels = ['', '3m', '6m', '9m', '12m']

    pprops = { 'xlim':        [np.min(xticks), np.max(xticks)],
                'xticks':      xticks,
                'xticklabels': xticklabels,
                'ylim':        [0, 1],
                'yticks':      [0, 0.5, 1],
                'yticklabels': ['', '', ''],
                'yminorticks': [0.25, 0.75],
                'axoffset':    0,
                'theme':       'boxed' }

    ax_clade_true.stackplot(x, Y, labels=clades, colors=colors, edgecolors='w', lw=AXWIDTH)
    mp.plot(type='line', ax=ax_clade_true, x=[[-1, -1]], y=[[-1, -1]], plotprops=DEF_LINEPROPS, **pprops)
    ax_clade_true.tick_params(axis='x', pad=4)

    ## null clade frequencies -- same as last observed in df_past

    x = df_fc['date'].unique()
    x.sort()
    Y = np.array([df_last[df_last['clade'] == clade]['frequency'].values[0] for clade in clades_to_plot])
    Y = np.tile(Y, (len(x), 1)).T

    xticks      = np.linspace(np.min(df_fc['date']), np.max(df_fc['date'])+1, num=5)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    # xticklabels = [xticklabels[i].strftime('%b %Y') if i in (0, len(xticklabels)-1) else '' for i in range(len(xticklabels))]
    xticklabels = ['', '3m', '6m', '9m', '12m']

    pprops = { 'xlim':        [np.min(xticks), np.max(xticks)],
                'xticks':      xticks,
                'xticklabels': xticklabels,
                'ylim':        [0, 1],
                'yticks':      [0, 0.5, 1],
                'yticklabels': ['', '', ''],
                'yminorticks': [0.25, 0.75],
                'axoffset':    0,
                'theme':       'boxed' }

    ax_clade_null.stackplot(x, Y, labels=clades, colors=colors, edgecolors='w', lw=AXWIDTH)
    mp.plot(type='line', ax=ax_clade_null, x=[[-1, -1]], y=[[-1, -1]], plotprops=DEF_LINEPROPS, **pprops)
    ax_clade_null.tick_params(axis='x', pad=4)


    # legend

    ax_clade_past.text(box_clade_past['left']-0.07, box_clade_past['top']+0.02, 'a'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_clade_fc.text(box_clade_fc['left']-0.07, box_clade_fc['top']+0.05, 'b'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_clade_true.text(box_clade_true['left']-0.02, box_clade_true['top']+0.05, 'c'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_clade_null.text(box_clade_null['left']-0.02, box_clade_null['top']+0.05, 'd'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    legend_x = box_clade_past['right'] + 0.02
    legend_y = box_clade_past['top'] - 0.06
    legend_dy = 0.04

    ax_clade_past.text(legend_x, legend_y + legend_dy, 'Clades:', ha='left', va='center', color=BKCOLOR, fontsize=SIZELABEL, transform=fig.transFigure, clip_on=False)
    for idx, clade in enumerate(clades):
        clade_color = colors[idx]
        ax_clade_past.text(legend_x, legend_y - idx * legend_dy, clade, ha='left', va='center', color=clade_color, fontweight='bold', fontsize=SIZELABEL, transform=fig.transFigure, clip_on=False)

    ax_clade_fc.text((box_clade_fc['left'] + box_clade_fc['right'])/2, box_clade_fc['top']+0.04 , 'Predicted (simulation)', ha='center', va='center', transform=fig.transFigure, **DEF_LABELPROPS)
    ax_clade_true.text((box_clade_true['left'] + box_clade_true['right'])/2, box_clade_true['top']+0.04 , 'Actual', ha='center', va='center', transform=fig.transFigure, **DEF_LABELPROPS)
    ax_clade_null.text((box_clade_null['left'] + box_clade_null['right'])/2, box_clade_null['top']+0.04 , 'Predicted (null)', ha='center', va='center', transform=fig.transFigure, **DEF_LABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_pfix(pfix_files, colors=None, fig_name='fig-pfix.pdf'):
    """ Curves showing probability that a mutation will fix, along with averages for
        top 50%/bottom 50% from simulations. """

    # Read in data
    df = [pd.read_csv(pfix_file, comment='#', memory_map=True) for pfix_file in pfix_files]

    labels = ['H3', 'N2', 'H1', 'N1']
    if not colors:
        colors = sns.husl_palette(len(df))
    
    # PLOT FIGURE
    
    ## set up figure grid

    w     = ONE_FIVE_COLUMN
    goldh = w
    fig   = plt.figure(figsize=(w, goldh))

    box = dict(left=0.08, right=0.95, top=0.95, bottom=0.08)
    gs  = gridspec.GridSpec(2, 2, **box)
    ax  = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])]

    lineprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 1.0 }
    dashprops = { 'lw': SIZELINE*1.0, 'linestyle': ':', 'alpha': 0.3 }

    for i in range(len(df)):
        pprops = { 'colors':      [C_BEN, C_BEN_LT, C_NEU, C_DEL_LT, C_DEL],
                   'xlim':        [0, 1.02],
                   'xticks':      [0, 0.5, 1],
                   'xticklabels': ['0', '0.5', '1'],
                   'ylim':        [0,  1.02],
                   'yticks':      [0, 0.5, 1],
                   'yticklabels': ['0', '0.5', '1'],
                   'axoffset':    0,
                   'hide':        ['top', 'right'],
                   'theme':       'open' }
        
        if i in [0, 1]:
            pprops['xticklabels'] = ['' for k in pprops['xticks']]
        if i in [1, 3]:
            pprops['yticklabels'] = ['' for k in pprops['yticks']]

        xs = [df[i]['Frequency'] for j in range(5)]
        ys = [df[i]['Top25_score'], df[i]['Top_score'], df[i]['Null'], df[i]['Bottom_score'], df[i]['Bottom25_score']]

        mp.plot(type='line', ax=ax[i], x=xs, y=ys, plotprops=lineprops, **pprops)

        pprops['colors'] = [BKCOLOR]
        mp.line(ax=ax[i], x=[[0, 1]], y=[[0, 1]], plotprops=dashprops, **pprops)

        # Legend/labels

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'center', 'va': 'center', 'clip_on': False}
        ax[i].text(0.05, 0.97, '%s' % labels[i], color=BKCOLOR, **txtprops)

        if i==1:
            txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'left', 'va': 'center', 'clip_on': False}
            legend_dx1 = -0.10
            legend_dx2 = -0.05
            legend_dy = -0.05
            legend_x =  0.70
            legend_y =  0.33
            legend_t = ['top 25%', 'top 50%', 'avg (all)', 'bottom 50%', 'bottom 25%']
            legend_c = [C_BEN, C_BEN_LT, C_NEU, C_DEL_LT, C_DEL]
            for k in range(len(legend_t)):
                # mp.line(ax=ax[i], x=[[legend_x + legend_dx1, legend_x + legend_dx2]],
                #         y=[[legend_y + (1.5 * k * legend_dy), legend_y + (1.5 * k * legend_dy)]],
                #         colors=[legend_c[k]], plotprops=dict(lw=1.5*SIZELINE, ls='-', clip_on=False))
                ax[i].text(legend_x, legend_y + (1.5 * k * legend_dy), legend_t[k], color=legend_c[k], fontweight='bold', **txtprops)
            ax[i].text(legend_x, legend_y + (-1.5 * legend_dy), 'Predicted to fix:', color=BKCOLOR, **txtprops)

    ax[0].text(box['left']-0.06, (box['top']+box['bottom'])/2, 'Fixation probability', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)
    ax[0].text((box['left']+box['right'])/2, box['bottom']-0.062, 'Allele frequency', transform=fig.transFigure, ha='center', va='center', **DEF_LABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_pfix_dauc(pfix_files, auc_files, colors=None, fig_name='fig-pfix-dauc.pdf'):
    """ Curves showing probability that a mutation will fix along with improvements in AUC vs. null. """

    # Read in data
    df = [pd.read_csv(pfix_file, comment='#', memory_map=True) for pfix_file in pfix_files]

    labels = ['H3', 'N2', 'H1', 'N1']
    if not colors:
        colors = sns.husl_palette(len(df))
    
    # PLOT FIGURE
    
    ## set up figure grid

    w     = DOUBLE_COLUMN
    goldh = w/2.8
    fig   = plt.figure(figsize=(w, goldh))

    box_l = 0.11
    box_r = 0.99
    box_t = 0.97
    box_b = 0.15

    dx_box = 0.05
    dy_box = 0.06
    x_pfix = (box_r - box_l - 3*dx_box) / 4
    y_pfix = (w/goldh) * x_pfix

    box_pfix = [dict(left=box_l+i*(x_pfix+dx_box), right=box_l+i*(x_pfix+dx_box)+x_pfix, top=box_t, bottom=box_t-y_pfix) for i in range(4)]
    box_dauc = [dict(left=box_l+i*(x_pfix+dx_box), right=box_l+i*(x_pfix+dx_box)+x_pfix, top=box_t-y_pfix-dy_box, bottom=box_b) for i in range(4)]

    gs_pfix  = [gridspec.GridSpec(1, 1, **box_pfix[i]) for i in range(4)]
    ax_pfix  = [plt.subplot(gs_pfix[i][0, 0]) for i in range(4)]
    gs_dauc  = [gridspec.GridSpec(1, 1, **box_dauc[i]) for i in range(4)]
    ax_dauc  = [plt.subplot(gs_dauc[i][0, 0]) for i in range(4)]

    # Probability of fixation curves

    lineprops = { 'lw': SIZELINE*1.2, 'linestyle': '-', 'alpha': 1.0 }
    dashprops = { 'lw': SIZELINE*1.0, 'linestyle': ':', 'alpha': 0.3 }

    for i in range(len(df)):
        pprops = { 'colors':      [C_BEN, C_BEN_LT, C_NEU, C_DEL_LT, C_DEL],
                   'xlim':        [0, 1.02],
                   'xticks':      [0, 0.5, 1],
                #    'xminorticks': [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
                #    'xticklabels': ['0', '0.5', '1'],
                   'xticklabels': ['' for k in [0, 0.5, 1]],
                   'ylim':        [0,  1.02],
                   'yticks':      [0, 0.5, 1],
                   'yticklabels': ['0', '0.5', '1'],
                #    'xlabel':      'Allele frequency',
                   'ylabel':      'Fixation probability',
                   'axoffset':    0,
                   'hide':        ['top', 'right'],
                   'theme':       'open' }
    
        if i!=0:
            pprops['ylabel'] = ''
            pprops['yticklabels'] = ['' for k in pprops['yticks']]

        xs = [df[i]['Frequency'] for j in range(5)]
        ys = [df[i]['Top25_score'], df[i]['Top_score'], df[i]['Null'], df[i]['Bottom_score'], df[i]['Bottom25_score']]

        mp.plot(type='line', ax=ax_pfix[i], x=xs, y=ys, plotprops=lineprops, **pprops)

        pprops['colors'] = [BKCOLOR]
        mp.line(ax=ax_pfix[i], x=[[0, 1]], y=[[0, 1]], plotprops=dashprops, **pprops)

        # Legend/labels

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'fontweight': 'bold', 'ha': 'center', 'va': 'center', 'clip_on': False}
        ax_pfix[i].text(0.070, -0.165, '%s' % labels[i], color=colors[i], **txtprops)

        if i==1:
            txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'left', 'va': 'center', 'clip_on': False}
            legend_dx1 = -0.10
            legend_dx2 = -0.05
            legend_dy = -0.06
            legend_x =  0.67
            legend_y =  0.44
            legend_t = ['top 25%', 'top 50%', 'avg (all)', 'bottom 50%', 'bottom 25%']
            legend_c = [C_BEN, C_BEN_LT, C_NEU, C_DEL_LT, C_DEL]
            for k in range(len(legend_t)):
                # mp.line(ax=ax[i], x=[[legend_x + legend_dx1, legend_x + legend_dx2]],
                #         y=[[legend_y + (1.5 * k * legend_dy), legend_y + (1.5 * k * legend_dy)]],
                #         colors=[legend_c[k]], plotprops=dict(lw=1.5*SIZELINE, ls='-', clip_on=False))
                ax_pfix[i].text(legend_x, legend_y + (1.5 * k * legend_dy), legend_t[k], color=legend_c[k], fontweight='bold', **txtprops)
            # ax_pfix[i].text(legend_x, legend_y + (-3 * legend_dy), 'Predicted', color=BKCOLOR, **txtprops)
            # ax_pfix[i].text(legend_x, legend_y + (-1.5 * legend_dy), 'to fix:', color=BKCOLOR, **txtprops)

    dx_sub = 0.095
    dy_sub = 0.035
    ax_pfix[0].text(box_pfix[0]['left']-dx_sub, box_pfix[0]['top']-dy_sub, 'a'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    # ax_pfix[0].text(box_pfix[0]['left']-0.07, (box_pfix[0]['top']+box_pfix[0]['bottom'])/2, 'Fixation probability', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)
    # ax_pfix[0].text((box_pfix[0]['left']+box_pfix[0]['right'])/2, box_pfix[0]['bottom']-0.072, 'Mutation frequency', transform=fig.transFigure, ha='center', va='center', **DEF_LABELPROPS)

    # Difference in AUC vs. null

    # Read in AUC data
    df = [pd.read_csv(auc_file, comment='#', memory_map=True) for auc_file in auc_files]

    plotprops = dict(lw=0, marker='o', s=2*SMALLSIZEDOT)
    dashlineprops = { 'lw': SIZELINE * 2.0, 'ls': ':', 'alpha': 0.5, 'color': BKCOLOR }

    for i in range(len(df)):
        df_auc = df[i]

        pprops = { 'xlim':        [0, 1.02],
                   'xticks':      [0, 0.5, 1],
                   'xticklabels': ['0', '0.5', '1'],
                #    'xminorticks': [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
                   'ylim':        [-0.5, 0.5],
                   'yticks':      [-0.5, 0.0, 0.5],
                   'xlabel':      'Allele frequency',
                #    'ylabel':      'Improvement\nvs. neutral model\n' + r'$(\Delta \mathrm{AUROC})$',
                   'plotprops':   plotprops,
                   'axoffset':    0,
                   'theme':       'open',
                   'hide':        ['top', 'right'] }

        if i!=0:
            pprops['ylabel'] = ''
            pprops['yticklabels'] = ['' for k in pprops['yticks']]

        mp.plot(type='scatter', ax=ax_dauc[i], x=[df_auc['Frequency']], y=[df_auc['AUC_simulation']-df_auc['AUC_null']], colors=[colors[i]], **pprops)
        ax_dauc[i].axhline(y=0, **dashlineprops)

    ax_dauc[0].text(box_dauc[0]['left']-dx_sub, box_dauc[0]['top']+dy_sub, 'b'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_dauc[0].text(box_dauc[0]['left']-0.063, (box_dauc[0]['top']+box_dauc[0]['bottom'])/2, 'Improvement\nvs. neutral model\n' + r'$(\Delta \mathrm{AUROC})$', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)
    # ax[0].text(box['left']-0.11, (box['top']+box['bottom'])/2, 'Accuracy of mutation fixation/loss prediction (AUROC)', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)
    # ax[2].text((box['left']+box['right'])/2, box['bottom']-0.06, 'Mutation frequency', transform=fig.transFigure, ha='center', va='center', **DEF_LABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_pfix_single(pfix_files, colors=None, fig_name='fig-pfix.pdf'):
    """ Curves showing probability that a mutation will fix, along with averages for
        top 50%/bottom 50% from simulations. """

    # Read in data
    df = [pd.read_csv(pfix_file, comment='#', memory_map=True) for pfix_file in pfix_files]

    labels = ['H3', 'N2', 'H1', 'N1']
    if not colors:
        colors = sns.husl_palette(len(df))
    
    # PLOT FIGURE
    
    ## set up figure grid

    w     = SINGLE_COLUMN
    goldh = w
    fig   = plt.figure(figsize=(w, goldh))

    box = dict(left=0.10, right=0.97, top=0.97, bottom=0.10)
    gs  = gridspec.GridSpec(2, 2, **box)
    ax  = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])]

    lineprops = { 'lw': SIZELINE*1.2, 'linestyle': '-', 'alpha': 1.0 }
    dashprops = { 'lw': SIZELINE*1.0, 'linestyle': ':', 'alpha': 0.3 }

    for i in range(len(df)):
        pprops = { 'colors':      [C_BEN, C_BEN_LT, C_NEU, C_DEL_LT, C_DEL],
                   'xlim':        [0, 1.02],
                   'xticks':      [0, 0.5, 1],
                   'xticklabels': ['0', '0.5', '1'],
                   'ylim':        [0,  1.02],
                   'yticks':      [0, 0.5, 1],
                   'yticklabels': ['0', '0.5', '1'],
                   'axoffset':    0,
                   'hide':        ['top', 'right'],
                   'theme':       'open' }
        
        if i in [0, 1]:
            pprops['xticklabels'] = ['' for k in pprops['xticks']]
        if i in [1, 3]:
            pprops['yticklabels'] = ['' for k in pprops['yticks']]

        xs = [df[i]['Frequency'] for j in range(5)]
        ys = [df[i]['Top25_score'], df[i]['Top_score'], df[i]['Null'], df[i]['Bottom_score'], df[i]['Bottom25_score']]

        mp.plot(type='line', ax=ax[i], x=xs, y=ys, plotprops=lineprops, **pprops)

        pprops['colors'] = [BKCOLOR]
        mp.line(ax=ax[i], x=[[0, 1]], y=[[0, 1]], plotprops=dashprops, **pprops)

        # Legend/labels

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'center', 'va': 'center', 'clip_on': False}
        ax[i].text(0.075, 0.97, '%s' % labels[i], color=BKCOLOR, **txtprops)

        if i==1:
            txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'left', 'va': 'center', 'clip_on': False}
            legend_dx1 = -0.10
            legend_dx2 = -0.05
            legend_dy = -0.05
            legend_x =  0.60
            legend_y =  0.33
            legend_t = ['top 25%', 'top 50%', 'avg (all)', 'bottom 50%', 'bottom 25%']
            legend_c = [C_BEN, C_BEN_LT, C_NEU, C_DEL_LT, C_DEL]
            for k in range(len(legend_t)):
                # mp.line(ax=ax[i], x=[[legend_x + legend_dx1, legend_x + legend_dx2]],
                #         y=[[legend_y + (1.5 * k * legend_dy), legend_y + (1.5 * k * legend_dy)]],
                #         colors=[legend_c[k]], plotprops=dict(lw=1.5*SIZELINE, ls='-', clip_on=False))
                ax[i].text(legend_x, legend_y + (1.5 * k * legend_dy), legend_t[k], color=legend_c[k], fontweight='bold', **txtprops)
            # ax[i].text(legend_x, legend_y + (-3 * legend_dy), 'Predicted', color=BKCOLOR, **txtprops)
            # ax[i].text(legend_x, legend_y + (-1.5 * legend_dy), 'to fix:', color=BKCOLOR, **txtprops)

    ax[0].text(box['left']-0.07, (box['top']+box['bottom'])/2, 'Fixation probability', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)
    ax[0].text((box['left']+box['right'])/2, box['bottom']-0.072, 'Mutation frequency', transform=fig.transFigure, ha='center', va='center', **DEF_LABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_AUC(auc_files, plot_null=True, plot_model=True, colors=None, fig_name='fig-auc.pdf'):
    """ AUC bar plots predicting fixation of mutations. """

    # Read in data
    df = [pd.read_csv(auc_file, comment='#', memory_map=True) for auc_file in auc_files]
    
    # PLOT FIGURE
    
    ## set up figure grid

    w     = SINGLE_COLUMN
    goldh = w * 1.5
    fig   = plt.figure(figsize=(w, goldh))

    box = dict(left=0.14, right=0.97, top=0.95, bottom=0.08)
    gs  = gridspec.GridSpec(2, 2, **box)
    ax  = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])]

    ## plot AUC

    histprops = dict(lw=AXWIDTH/2, width=0.9*0.1, align='center', orientation='vertical', edgecolor=[BKCOLOR], alpha=0.75)
    
    labels = ['H3', 'N2', 'H1', 'N1']

    for i in range(len(df)):
        df_auc = df[i]

        pprops = { 'xlim':        [0, 1],
                   'xticks':      [0, 0.5, 1],
                   'xticklabels': ['0', '0.5', '1'],
                   'xminorticks': [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
                   'ylim':        [0., 1],
                   'yticks':      [0., 0.5, 1],
                   'yticklabels': ['0', '0.5', '1'],
                   'yminorticks': [0.25, 0.75],
                   'plotprops':   histprops,
                   'theme':       'open',
                   'hide':        ['top', 'right'] }

        if i in [0, 1]:
            pprops['xticklabels'] = ['' for k in pprops['xticks']]
        if i in [1, 3]:
            pprops['yticklabels'] = ['' for k in pprops['yticks']]

        if plot_model:
            pprops['colors'] = [colors[i]] if colors else [C_MPL]
            mp.bar(ax=ax[i], x=[df_auc['Frequency']], y=[df_auc['AUC_simulation']], **pprops)

            print('%s\t%.2f' % (labels[i], np.mean(df_auc['AUC_simulation']-df_auc['AUC_null'])))

        if plot_null:
            pprops['colors'] = [C_NEU]
            mp.plot(type='bar', ax=ax[i], x=[df_auc['Frequency']], y=[df_auc['AUC_null']], **pprops)

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'fontweight': 'bold', 'ha': 'right', 'va': 'center', 'clip_on': False}
        ax[i].text(0.95, 0.97, '%s model' % labels[i], color=colors[i], **txtprops)
        # ax[i].text(0.95, 0.97, '%s single site' % labels[i], color=colors[i], **txtprops)
        ax[i].text(0.95, 0.90, 'null', color=C_NEU, **txtprops)

    ax[0].text(box['left']-0.11, (box['top']+box['bottom'])/2, 'Accuracy of mutation fixation/loss prediction (AUROC)', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)
    ax[2].text((box['left']+box['right'])/2, box['bottom']-0.06, 'Mutation frequency', transform=fig.transFigure, ha='center', va='center', **DEF_LABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_dw(dw_files, t_min=0, dt_min=0, colors=None, fig_name='fig-dw.pdf', **kwargs):
    """ Plot difference in background statistics for mutations that fix vs. those that are lost. """

    # Read in data
    df = [pd.read_csv(dw_file, comment='#', memory_map=True) for dw_file in dw_files]

    # Clip mutations that are too early or too short-lived
    for i in range(len(df)):
        if t_min>0 and 'T0' in df[i]:
            df[i] = df[i][df[i]['T0']>t_min]
        if dt_min>0 and 'dT' in df[i]:
            df[i] = df[i][(df[i]['T1']-df[i]['T0'])>dt_min]

    labels = ['H3', 'N2', 'H1', 'N1']
    if not colors:
        colors = sns.husl_palette(len(df))
    
    # PLOT FIGURE
    
    ## set up figure grid

    w     = SINGLE_COLUMN
    goldh = w
    fig   = plt.figure(figsize=(w, goldh))

    box_l = 0.22
    box_r = 0.90
    box_t = 0.90
    box_b = 0.03

    dy_box = 0.07
    y_w = (box_t - box_b - 2*dy_box) / 3

    box_ds      = dict(left=box_l, right=box_r, top=box_t,                bottom=box_t-y_w)
    box_dw_bg   = dict(left=box_l, right=box_r, top=box_t-1*(y_w+dy_box), bottom=box_t-1*(y_w+dy_box)-y_w)
    box_dw_rate = dict(left=box_l, right=box_r, top=box_t-2*(y_w+dy_box), bottom=box_t-2*(y_w+dy_box)-y_w)

    gs_ds      = gridspec.GridSpec(1, 4, **box_ds)
    ax_ds      = [plt.subplot(gs_ds[0, i]) for i in range(4)]
    gs_dw_bg   = gridspec.GridSpec(1, 4, **box_dw_bg)
    ax_dw_bg   = [plt.subplot(gs_dw_bg[0, i]) for i in range(4)]
    gs_dw_rate = gridspec.GridSpec(1, 4, **box_dw_rate)
    ax_dw_rate = [plt.subplot(gs_dw_rate[0, i]) for i in range(4)]

    scatterprops = dict(lw=0, s=SMALLSIZEDOT, marker='o', alpha=0.25)
    errorprops   = dict(lw=AXWIDTH, markersize=SMALLSIZEDOT, fmt='.', elinewidth=AXWIDTH, capthick=0, capsize=0, mew=AXWIDTH, alpha=1)

    for k in range(len(df)):

        df_dw = df[k]

        # Get unique mutation names
        name_vec = []
        for it, row in df_dw.iterrows():
            name_vec.append(str(row['Position'])+'-'+str(row['Nucleotide']))
        df_dw['name'] = name_vec
        names = np.unique(df_dw['name'])

        # Get dw for fixed and lost mutations
        ds_fixed = []
        ds_lost = []
        dw_fixed = []
        dw_lost = []
        dt_fixed = []
        dt_lost = []
        x_std  = 0.1
        x_fixed = []
        x_lost = []
        for name in names:
            row_name = df_dw[df_dw['name']==name].iloc[0]
            dt_temp = row_name['T1'] - row_name['T0']
            if dt_temp<dt_min or row_name['T0']<t_min:
                continue
            if row_name['Fixation']=='fixed':
                ds_fixed.append(row_name['fitness'])
                dw_fixed.append(row_name['w1_bg'] - row_name['w0_bg'])
                dt_fixed.append(dt_temp)
                x_fixed.append(np.random.normal(0.5, x_std))
            elif row_name['Fixation']=='lost':
                ds_lost.append(row_name['fitness'])
                dw_lost.append(row_name['w1_bg'] - row_name['w0_bg'])
                dt_lost.append(dt_temp)
                x_lost.append(np.random.normal(1.5, x_std))

        ds_fixed = np.array(ds_fixed)
        ds_lost = np.array(ds_lost)
        dw_fixed = np.array(dw_fixed)
        dw_lost = np.array(dw_lost)
        dt_fixed = np.array(dt_fixed)
        dt_lost = np.array(dt_lost)

        pprops = { 'colors':      [colors[k], C_NEU_LT],
                   'xlim':        [0, 2],
                   'xticks':      [],
                   'yticks':      [],
                   'ylabel':      '',
                   'axoffset':    0,
                   'hide':        ['top', 'right', 'left', 'bottom'],
                   'theme':       'open' }

        # print('%s' % labels[k])
        # print('ds: %.2e, %.2e' % (np.min([np.min(ds_fixed), np.min(ds_lost)]), np.max([np.max(ds_fixed), np.max(ds_lost)])))

        # t, p = st.ttest_ind(dw_fixed, dw_lost, equal_var=False)
        # ax[i].text(-0.018, 0.35, 'p=%.1e' % p, color=BKCOLOR, **txtprops)

        pprops['ylim'] = [-0.006, 0.011]

        if k==0:
            pprops['ylabel'] = 'Selection\ncoefficient (%)'
            pprops['yticks'] = [-0.005, 0, 0.005, 0.01]
            pprops['yticklabels'] = ['-0.5', '0', '0.5', '1']
            pprops['hide'] = ['top', 'right', 'bottom']

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'center', 'va': 'center', 'clip_on': False}
        if k==0:
            ax_ds[k].text(0.5, 0.012, 'fixed', color=colors[k], fontweight='bold', **txtprops)
            ax_ds[k].text(1.5, 0.009, 'lost',   color=C_NEU, fontweight='bold', **txtprops)
        
        ax_ds[k].text(1.0, 0.015, '%s' % labels[k], color=BKCOLOR, **txtprops)

        t, p = st.ttest_ind(ds_fixed, ds_lost, equal_var=False)
        if p<1e-3:
            ax_ds[k].text(1.0, 0.011, '***', color=BKCOLOR, **txtprops)
        elif p<1e-2:
            ax_ds[k].text(1.0, 0.011, '**', color=BKCOLOR, **txtprops)
        elif p<0.05:
            ax_ds[k].text(1.0, 0.011, '*', color=BKCOLOR, **txtprops)

        mp.scatter(ax=ax_ds[k], x=[x_fixed, x_lost], y=[ds_fixed, ds_lost], plotprops=scatterprops, **pprops)
        mp.plot(type='error', ax=ax_ds[k], x=[[0.5], [1.5]], y=[[np.mean(ds_fixed)], [np.mean(ds_lost)]], yerr=[[np.std(ds_fixed)], [np.std(ds_lost)]],
                edgecolor=[BKCOLOR, BKCOLOR], facecolor=[colors[k], C_NEU_LT], plotprops=errorprops, **pprops)

        # print('dw: %.2e, %.2e' % (np.min([np.min(dw_fixed), np.min(dw_lost)]), np.max([np.max(dw_fixed), np.max(dw_lost)])))

        pprops['ylim'] = [-0.01, 0.034]

        if k==0:
            pprops['ylabel'] = 'Background\nfitness\nadvantage (%)'
            pprops['yticks'] = [-0.01, 0, 0.01, 0.02, 0.03]
            pprops['yticklabels'] = ['-1', '0', '1', '2', '3']

        t, p = st.ttest_ind(dw_fixed, dw_lost, equal_var=False)
        if p<1e-3:
            ax_dw_bg[k].text(1.0, 0.034, '***', color=BKCOLOR, **txtprops)
        elif p<1e-2:
            ax_dw_bg[k].text(1.0, 0.034, '**', color=BKCOLOR, **txtprops)
        elif p<0.05:
            ax_dw_bg[k].text(1.0, 0.034, '*', color=BKCOLOR, **txtprops)

        mp.scatter(ax=ax_dw_bg[k], x=[x_fixed, x_lost], y=[dw_fixed, dw_lost], plotprops=scatterprops, **pprops)
        mp.plot(type='error', ax=ax_dw_bg[k], x=[[0.5], [1.5]], y=[[np.mean(dw_fixed)], [np.mean(dw_lost)]], yerr=[[np.std(dw_fixed)], [np.std(dw_lost)]],
                edgecolor=[BKCOLOR, BKCOLOR], facecolor=[colors[k], C_NEU_LT], plotprops=errorprops, **pprops)

        # print('dw/dt: %.2e, %.2e\n' % (365*np.min([np.min(dw_fixed/dt_fixed), np.min(dw_lost/dt_lost)]), 365*np.max([np.max(dw_fixed/dt_fixed), np.max(dw_lost/dt_lost)])))

        pprops['ylim'] = [-0.022, 0.03]
        
        if k==0:
            pprops['ylabel'] = 'Background\nfitness growth\n(%/year)'
            pprops['yticks'] = [-0.02, -0.01, 0, 0.01, 0.02, 0.03]
            pprops['yticklabels'] = ['-2', '-1', '0', '1', '2', '3']

        t, p = st.ttest_ind(dw_fixed/dt_fixed, dw_lost/dt_lost, equal_var=False)
        if p<1e-3:
            ax_dw_rate[k].text(1.0, 0.03, '***', color=BKCOLOR, **txtprops)
        elif p<1e-2:
            ax_dw_rate[k].text(1.0, 0.03, '**', color=BKCOLOR, **txtprops)
        elif p<0.05:
            ax_dw_rate[k].text(1.0, 0.03, '*', color=BKCOLOR, **txtprops)

        mp.scatter(ax=ax_dw_rate[k], x=[x_fixed, x_lost], y=[365*dw_fixed/dt_fixed, 365*dw_lost/dt_lost], plotprops=scatterprops, **pprops)
        mp.plot(type='error', ax=ax_dw_rate[k], x=[[0.5], [1.5]], y=[[np.mean(365*dw_fixed/dt_fixed)], [np.mean(365*dw_lost/dt_lost)]], yerr=[[np.std(365*dw_fixed/dt_fixed)], [np.std(365*dw_lost/dt_lost)]],
                edgecolor=[BKCOLOR, BKCOLOR], facecolor=[colors[k], C_NEU_LT], plotprops=errorprops, **pprops)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_clade(df_past, df_full, df_forecast, colors=None, fig_name='fig-clade.pdf', **kwargs):
    """ Illustrative plot of clade dynamics. """
    
    # Read in clade data
    df_fc = pd.concat(df_forecast)
    df_past['date'] = pd.to_datetime(df_past['date'])
    df_full['date'] = pd.to_datetime(df_full['date'])
    df_fc['date'] = pd.to_datetime(df_fc['date'])
    df_past['date'] = (df_past['date'] - dt.datetime(2009, 1, 1)).dt.days
    df_full['date'] = (df_full['date'] - dt.datetime(2009, 1, 1)).dt.days
    df_fc['date'] = (df_fc['date'] - dt.datetime(2009, 1, 1)).dt.days

    df_past['clade'] = df_past['clade'].astype(str).str.strip()
    df_full['clade'] = df_full['clade'].astype(str).str.strip()
    df_fc['clade'] = df_fc['clade'].astype(str).str.strip()

    df_past['frequency'] = df_past.groupby('date')['frequency'].transform(lambda x: x / x.sum())
    df_full['frequency'] = df_full.groupby('date')['frequency'].transform(lambda x: x / x.sum())
    df_fc['frequency'] = df_fc.groupby('date')['frequency'].transform(lambda x: x / x.sum())

    pivot_past = df_past.pivot_table(index='date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot_past = pivot_past.sort_index()
    pivot_full = df_full.pivot_table(index='date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot_full = pivot_full.sort_index()
    pivot_fc = df_fc.pivot_table(index='date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot_fc = pivot_fc.sort_index()

    label_threshold = 0.01
    max_freqs = pivot_past.max()
    clades_to_plot = max_freqs[max_freqs > label_threshold].index
    pivot_past = pivot_past[clades_to_plot]
    pivot_full = pivot_full[clades_to_plot]
    pivot_fc = pivot_fc[clades_to_plot]

    # PLOT FIGURE
    
    ## set up figure grid

    w     = SINGLE_COLUMN
    goldh = w*1.2
    fig   = plt.figure(figsize=(w, goldh))

    box_top = 0.92
    box_bot = 0.30
    box_lft = 0.12
    box_rt  = 0.90
    box_dy  = 0.09
    box_dx  = 0.16
    box_x  = (box_rt - box_lft - box_dx) / 2

    past_len = df_past['date'].max() - df_past['date'].min()
    fc_min = df_past['date'].max()
    fc_max = df_fc['date'].max()
    fc_len = fc_max - fc_min

    total_dy_past = box_top - box_bot
    total_dy_fc = 3 * (fc_len / past_len) * total_dy_past + 2 * box_dy
    if total_dy_fc > 1.0:
        print('Need to resize past box or reduce spacing to fit forecast boxes: total_dy_fc = %.2f' % total_dy_fc)

    fc_dy = total_dy_past * (fc_len / past_len)

    box_clade_past = dict(left=box_lft,              right=box_lft+box_x, top=box_top,                    bottom=box_bot)
    box_clade_fc   = dict(left=box_lft+box_x+box_dx, right=box_rt,        top=box_top-(0*fc_dy+0*box_dy), bottom=box_top-(1*fc_dy+0*box_dy))
    box_clade_true = dict(left=box_lft+box_x+box_dx, right=box_rt,        top=box_top-(1*fc_dy+1*box_dy), bottom=box_top-(2*fc_dy+1*box_dy))
    box_clade_null = dict(left=box_lft+box_x+box_dx, right=box_rt,        top=box_top-(2*fc_dy+2*box_dy), bottom=box_top-(3*fc_dy+2*box_dy))

    gs_clade_past = gridspec.GridSpec(1, 1, **box_clade_past)
    ax_clade_past = plt.subplot(gs_clade_past[0, 0])
    gs_clade_fc   = gridspec.GridSpec(1, 1, **box_clade_fc)
    ax_clade_fc   = plt.subplot(gs_clade_fc[0, 0])
    gs_clade_true = gridspec.GridSpec(1, 1, **box_clade_true)
    ax_clade_true = plt.subplot(gs_clade_true[0, 0])
    gs_clade_null = gridspec.GridSpec(1, 1, **box_clade_null)
    ax_clade_null = plt.subplot(gs_clade_null[0, 0])
    
    ## past clade frequencies

    x = pivot_past.index
    Y = pivot_past.values.T
    clades = pivot_past.columns

    # colors = sns.husl_palette(len(clades))
    # colors = [k for k in sns.color_palette('Blues', n_colors=2)] + [C_BEN] + [k for k in sns.color_palette('Greens', n_colors=3)]
    colors = sns.color_palette("ch:s=-.5,r=.6", n_colors=len(clades))
    # colors = sns.cubehelix_palette(n_colors=len(clades))
    # colors = ['#3B76AF', '#B3C7E5', '#EF8636', '#BCBD46', '#DBDB96', '#ABD8E3']

    xticks      = np.linspace(np.min(df_past['date']), np.max(df_past['date'])+1, num=5)
    xlim        = [np.min(xticks), np.max(xticks)][::-1]
    # xticks[-1]  = xticks[-1] - 50 # shift slightly to avoid text overlap
    # xticks[-2]  = xticks[-2] - 33.33
    # xticks[-3]  = xticks[-3] - 16.67
    xticks      = xticks[:-1] # remove last tick to avoid overlap with text
    dx_tick     = (xticks[1] - xticks[0]) / 2
    xticks      = xticks + dx_tick # shift ticks to avoid overlap
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    xticklabels = ['\n'.join(i.strftime('%b %Y').split()) for i in xticklabels]

    # fillprops = dict(lw=AXWIDTH, ec='w')
    fillprops = dict(lw=0, ec='None')

    pprops = { 'ylim':        xlim,
               'yticks':      xticks,
               'yticklabels': xticklabels,
               'xlim':        [0, 1],
               'xticks':      [0, 0.5, 1],
               'xticklabels': ['', '', ''],
               'xminorticks': [0.25, 0.75],
               'xlabel':      'Clade frequency',
               'axoffset':    0,
               'theme':       'boxed' }
    
    cumsum = np.zeros(len(x))
    for k in range(len(clades)):
        ax_clade_past.fill_betweenx(x, cumsum, cumsum + Y[k], fc=colors[k], **fillprops)
        cumsum += Y[k]

    mp.plot(type='line', ax=ax_clade_past, x=[[-1, -1]], y=[[-1, -1]], plotprops=DEF_LINEPROPS, **pprops)

    txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'center', 'va': 'center', 'clip_on': False}
    ax_clade_past.text((box_clade_past['left']+box_clade_past['right'])/2, box_clade_past['top']+0.03, 'Past evolution', color=BKCOLOR, transform=fig.transFigure, **txtprops)
    
    ## forecast clade frequencies

    df_last = df_past[df_past['date'] == df_past['date'].max()]
    df_fc = pd.concat([df_last, df_fc])

    pivot_fc = df_fc.pivot_table(index='date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot_fc = pivot_fc.sort_index()
    pivot_fc = pivot_fc[clades_to_plot]
    x = pivot_fc.index
    Y = pivot_fc.values.T

    xticks      = np.linspace(np.min(df_fc['date']), np.max(df_fc['date'])+1, num=5)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    xticklabels = ['', '3m', '6m', '9m', '12m']

    pprops = { 'ylim':        [np.min(xticks), np.max(xticks)][::-1],
               'yticks':      xticks,
               'yticklabels': xticklabels,
               'xlim':        [0, 1],
               'xticks':      [0, 0.5, 1],
               'xticklabels': ['', '', ''],
               'xminorticks': [0.25, 0.75],
            #    'xlabel':      'Clade frequency',
               'axoffset':    0,
               'theme':       'boxed' }

    cumsum = np.zeros(len(x))
    for k in range(len(clades)):
        ax_clade_fc.fill_betweenx(x, cumsum, cumsum + Y[k], fc=colors[k], **fillprops)
        cumsum += Y[k]

    mp.plot(type='line', ax=ax_clade_fc, x=[[-1, -1]], y=[[-1, -1]], plotprops=DEF_LINEPROPS, **pprops)

    txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'center', 'va': 'center', 'clip_on': False}
    ax_clade_fc.text((box_clade_fc['left']+box_clade_fc['right'])/2, box_clade_fc['top']+0.03, 'Model forecast', color=BKCOLOR, transform=fig.transFigure, **txtprops)
    
    ## true clade frequencies

    pivot_true = df_full.pivot_table(index='date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot_true = pivot_true.sort_index()
    pivot_true = pivot_true[clades_to_plot]
    x = pivot_true.index
    Y = pivot_true.values.T

    xticks      = np.linspace(np.min(df_fc['date']), np.max(df_fc['date'])+1, num=5)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    xticklabels = ['', '3m', '6m', '9m', '12m']

    pprops = { 'ylim':        [np.min(xticks), np.max(xticks)][::-1],
               'yticks':      xticks,
               'yticklabels': xticklabels,
               'xlim':        [0, 1],
               'xticks':      [0, 0.5, 1],
               'xticklabels': ['', '', ''],
               'xminorticks': [0.25, 0.75],
               'axoffset':    0,
               'theme':       'boxed' }

    cumsum = np.zeros(len(x))
    for k in range(len(clades)):
        ax_clade_true.fill_betweenx(x, cumsum, cumsum + Y[k], fc=colors[k], **fillprops)
        cumsum += Y[k]

    mp.plot(type='line', ax=ax_clade_true, x=[[-1, -1]], y=[[-1, -1]], plotprops=DEF_LINEPROPS, **pprops)

    txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'center', 'va': 'center', 'clip_on': False}
    ax_clade_true.text((box_clade_true['left']+box_clade_true['right'])/2, box_clade_true['top']+0.03, 'True frequencies', color=BKCOLOR, transform=fig.transFigure, **txtprops)
    
    ## null clade frequencies -- same as last observed in df_past

    x = df_fc['date'].unique()
    x.sort()
    Y = np.array([df_last[df_last['clade'] == clade]['frequency'].values[0] for clade in clades_to_plot])
    Y = np.tile(Y, (len(x), 1)).T

    xticks      = np.linspace(np.min(df_fc['date']), np.max(df_fc['date'])+1, num=5)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    xticklabels = ['', '3m', '6m', '9m', '12m']

    pprops = { 'ylim':        [np.min(xticks), np.max(xticks)][::-1],
               'yticks':      xticks,
               'yticklabels': xticklabels,
               'xlim':        [0, 1],
               'xticks':      [0, 0.5, 1],
               'xticklabels': ['', '', ''],
               'xminorticks': [0.25, 0.75],
               'xlabel':      'Clade frequency',
               'axoffset':    0,
               'theme':       'boxed' }

    cumsum = np.zeros(len(x))
    for k in range(len(clades)):
        ax_clade_null.fill_betweenx(x, cumsum, cumsum + Y[k], fc=colors[k], **fillprops)
        cumsum += Y[k]

    mp.plot(type='line', ax=ax_clade_null, x=[[-1, -1]], y=[[-1, -1]], plotprops=DEF_LINEPROPS, **pprops)

    txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'center', 'va': 'center', 'clip_on': False}
    ax_clade_null.text((box_clade_null['left']+box_clade_null['right'])/2, box_clade_null['top']+0.03, 'Neutral forecast', color=BKCOLOR, transform=fig.transFigure, **txtprops)

    # legend

    ax_clade_past.text(box_clade_past['left']-0.06, box_clade_past['top']+0.025, 'a'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_clade_fc.text(box_clade_fc['left']-0.06, box_clade_fc['top']+0.025, 'b'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_clade_true.text(box_clade_true['left']-0.06, box_clade_true['top']+0.025, 'c'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_clade_null.text(box_clade_null['left']-0.06, box_clade_null['top']+0.025, 'd'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    legend_x = box_clade_past['left'] - 0.00
    legend_y = box_clade_past['bottom'] - 0.10
    legend_dy = 0.03
    legend_dx = 0.12

    # ax_clade_past.text(legend_x, legend_y + legend_dy, 'Clades:', ha='left', va='center', color=BKCOLOR, fontsize=SIZELABEL, transform=fig.transFigure, clip_on=False)
    for idx, clade in enumerate(clades[:4]):
        clade_color = colors[idx]
        ax_clade_past.text(legend_x, legend_y - idx * legend_dy, clade, ha='left', va='center', color=clade_color, fontweight='bold', fontsize=SIZELABEL, transform=fig.transFigure, clip_on=False)

    for idx, clade in enumerate(clades[4:8]):
        clade_color = colors[idx + 4]
        ax_clade_past.text(legend_x + legend_dx, legend_y - idx * legend_dy, clade, ha='left', va='center', color=clade_color, fontweight='bold', fontsize=SIZELABEL, transform=fig.transFigure, clip_on=False)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_dw_clade(dw_files, df_past, df_full, df_forecast, t_min=0, dt_min=0, colors=None, fig_name='fig-dw-clade.pdf'):
    """ Plot difference in background statistics for mutations that fix vs. those that are lost,
        include illustrative plot of clade dynamics. """

    # Read in data
    df = [pd.read_csv(dw_file, comment='#', memory_map=True) for dw_file in dw_files]

    # Clip mutations that are too early or too short-lived
    for i in range(len(df)):
        if t_min>0 and 'T0' in df[i]:
            df[i] = df[i][df[i]['T0']>t_min]
        if dt_min>0 and 'dT' in df[i]:
            df[i] = df[i][(df[i]['T1']-df[i]['T0'])>dt_min]

    labels = ['H3', 'N2', 'H1', 'N1']
    if not colors:
        colors = sns.husl_palette(len(df))
    
    # PLOT FIGURE
    
    ## set up figure grid

    w     = ONE_FIVE_COLUMN
    goldh = w/1.4
    fig   = plt.figure(figsize=(w, goldh))

    box_l = 0.14
    box_r = 0.50
    box_t = 0.90
    box_b = 0.03

    dx_box = 0.05
    dy_box = 0.06
    y_w = (box_t - box_b - 2*dy_box) / 3

    box_ds      = dict(left=box_l, right=box_r, top=box_t,                bottom=box_t-y_w)
    box_dw_bg   = dict(left=box_l, right=box_r, top=box_t-1*(y_w+dy_box), bottom=box_t-1*(y_w+dy_box)-y_w)
    box_dw_rate = dict(left=box_l, right=box_r, top=box_t-2*(y_w+dy_box), bottom=box_t-2*(y_w+dy_box)-y_w)

    gs_ds      = gridspec.GridSpec(1, 4, **box_ds)
    ax_ds      = [plt.subplot(gs_ds[0, i]) for i in range(4)]
    gs_dw_bg   = gridspec.GridSpec(1, 4, **box_dw_bg)
    ax_dw_bg   = [plt.subplot(gs_dw_bg[0, i]) for i in range(4)]
    gs_dw_rate = gridspec.GridSpec(1, 4, **box_dw_rate)
    ax_dw_rate = [plt.subplot(gs_dw_rate[0, i]) for i in range(4)]

    scatterprops = dict(lw=0, s=SMALLSIZEDOT, marker='o', alpha=0.25)
    errorprops   = dict(lw=AXWIDTH, markersize=SMALLSIZEDOT, fmt='.', elinewidth=AXWIDTH, capthick=0, capsize=0, mew=AXWIDTH, alpha=1)

    for k in range(len(df)):

        df_dw = df[k]

        # Get unique mutation names
        name_vec = []
        for it, row in df_dw.iterrows():
            name_vec.append(str(row['Position'])+'-'+str(row['Nucleotide']))
        df_dw['name'] = name_vec
        names = np.unique(df_dw['name'])

        # Get dw for fixed and lost mutations
        ds_fixed = []
        ds_lost = []
        dw_fixed = []
        dw_lost = []
        dt_fixed = []
        dt_lost = []
        x_std  = 0.1
        x_fixed = []
        x_lost = []
        for name in names:
            row_name = df_dw[df_dw['name']==name].iloc[0]
            dt_temp = row_name['T1'] - row_name['T0']
            if dt_temp<dt_min or row_name['T0']<t_min:
                continue
            if row_name['Fixation']=='fixed':
                ds_fixed.append(row_name['fitness'])
                dw_fixed.append(row_name['w1_bg'] - row_name['w0_bg'])
                dt_fixed.append(dt_temp)
                x_fixed.append(np.random.normal(0.5, x_std))
            elif row_name['Fixation']=='lost':
                ds_lost.append(row_name['fitness'])
                dw_lost.append(row_name['w1_bg'] - row_name['w0_bg'])
                dt_lost.append(dt_temp)
                x_lost.append(np.random.normal(1.5, x_std))

        ds_fixed = np.array(ds_fixed)
        ds_lost = np.array(ds_lost)
        dw_fixed = np.array(dw_fixed)
        dw_lost = np.array(dw_lost)
        dt_fixed = np.array(dt_fixed)
        dt_lost = np.array(dt_lost)

        pprops = { 'colors':      [colors[k], C_NEU_LT],
                   'xlim':        [0, 2],
                   'xticks':      [],
                   'yticks':      [],
                   'ylabel':      '',
                   'axoffset':    0,
                   'hide':        ['top', 'right', 'left', 'bottom'],
                   'theme':       'open' }

        # print('%s' % labels[k])
        # print('ds: %.2e, %.2e' % (np.min([np.min(ds_fixed), np.min(ds_lost)]), np.max([np.max(ds_fixed), np.max(ds_lost)])))

        # t, p = st.ttest_ind(dw_fixed, dw_lost, equal_var=False)
        # ax[i].text(-0.018, 0.35, 'p=%.1e' % p, color=BKCOLOR, **txtprops)

        pprops['ylim'] = [-0.006, 0.011]

        if k==0:
            pprops['ylabel'] = 'Selection\ncoefficient (%)'
            pprops['yticks'] = [-0.005, 0, 0.005, 0.01]
            pprops['yticklabels'] = ['-0.5', '0', '0.5', '1']
            pprops['hide'] = ['top', 'right', 'bottom']

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'center', 'va': 'center', 'clip_on': False}
        if k==0:
            ax_ds[k].text(0.5, 0.012, 'fixed', color=colors[k], fontweight='bold', **txtprops)
            ax_ds[k].text(1.5, 0.009, 'lost',   color=C_NEU, fontweight='bold', **txtprops)
        
        ax_ds[k].text(1.0, 0.015, '%s' % labels[k], color=BKCOLOR, **txtprops)

        t, p = st.ttest_ind(ds_fixed, ds_lost, equal_var=False)
        if p<1e-3:
            ax_ds[k].text(1.0, 0.011, '***', color=BKCOLOR, **txtprops)
        elif p<1e-2:
            ax_ds[k].text(1.0, 0.011, '**', color=BKCOLOR, **txtprops)
        elif p<0.05:
            ax_ds[k].text(1.0, 0.011, '*', color=BKCOLOR, **txtprops)

        mp.scatter(ax=ax_ds[k], x=[x_fixed, x_lost], y=[ds_fixed, ds_lost], plotprops=scatterprops, **pprops)
        mp.plot(type='error', ax=ax_ds[k], x=[[0.5], [1.5]], y=[[np.mean(ds_fixed)], [np.mean(ds_lost)]], yerr=[[np.std(ds_fixed)], [np.std(ds_lost)]],
                edgecolor=[BKCOLOR, BKCOLOR], facecolor=[colors[k], C_NEU_LT], plotprops=errorprops, **pprops)

        # print('dw: %.2e, %.2e' % (np.min([np.min(dw_fixed), np.min(dw_lost)]), np.max([np.max(dw_fixed), np.max(dw_lost)])))

        pprops['ylim'] = [-0.01, 0.034]

        if k==0:
            pprops['ylabel'] = 'Background\nfitness\nadvantage (%)'
            pprops['yticks'] = [-0.01, 0, 0.01, 0.02, 0.03]
            pprops['yticklabels'] = ['-1', '0', '1', '2', '3']

        t, p = st.ttest_ind(dw_fixed, dw_lost, equal_var=False)
        if p<1e-3:
            ax_dw_bg[k].text(1.0, 0.034, '***', color=BKCOLOR, **txtprops)
        elif p<1e-2:
            ax_dw_bg[k].text(1.0, 0.034, '**', color=BKCOLOR, **txtprops)
        elif p<0.05:
            ax_dw_bg[k].text(1.0, 0.034, '*', color=BKCOLOR, **txtprops)

        mp.scatter(ax=ax_dw_bg[k], x=[x_fixed, x_lost], y=[dw_fixed, dw_lost], plotprops=scatterprops, **pprops)
        mp.plot(type='error', ax=ax_dw_bg[k], x=[[0.5], [1.5]], y=[[np.mean(dw_fixed)], [np.mean(dw_lost)]], yerr=[[np.std(dw_fixed)], [np.std(dw_lost)]],
                edgecolor=[BKCOLOR, BKCOLOR], facecolor=[colors[k], C_NEU_LT], plotprops=errorprops, **pprops)

        # print('dw/dt: %.2e, %.2e\n' % (365*np.min([np.min(dw_fixed/dt_fixed), np.min(dw_lost/dt_lost)]), 365*np.max([np.max(dw_fixed/dt_fixed), np.max(dw_lost/dt_lost)])))

        pprops['ylim'] = [-0.022, 0.03]
        
        if k==0:
            pprops['ylabel'] = 'Background\nfitness growth\n(%/year)'
            pprops['yticks'] = [-0.02, -0.01, 0, 0.01, 0.02, 0.03]
            pprops['yticklabels'] = ['-2', '-1', '0', '1', '2', '3']

        t, p = st.ttest_ind(dw_fixed/dt_fixed, dw_lost/dt_lost, equal_var=False)
        if p<1e-3:
            ax_dw_rate[k].text(1.0, 0.03, '***', color=BKCOLOR, **txtprops)
        elif p<1e-2:
            ax_dw_rate[k].text(1.0, 0.03, '**', color=BKCOLOR, **txtprops)
        elif p<0.05:
            ax_dw_rate[k].text(1.0, 0.03, '*', color=BKCOLOR, **txtprops)

        mp.scatter(ax=ax_dw_rate[k], x=[x_fixed, x_lost], y=[365*dw_fixed/dt_fixed, 365*dw_lost/dt_lost], plotprops=scatterprops, **pprops)
        mp.plot(type='error', ax=ax_dw_rate[k], x=[[0.5], [1.5]], y=[[np.mean(365*dw_fixed/dt_fixed)], [np.mean(365*dw_lost/dt_lost)]], yerr=[[np.std(365*dw_fixed/dt_fixed)], [np.std(365*dw_lost/dt_lost)]],
                edgecolor=[BKCOLOR, BKCOLOR], facecolor=[colors[k], C_NEU_LT], plotprops=errorprops, **pprops)

    # Read in clade data
    for df in [df_past, df_full] + df_forecast:
        df['date'] = pd.to_datetime(df['date'])
        # convert date to days since 2009-01-01
        df['date'] = (df['date'] - dt.datetime(2009, 1, 1)).dt.days

    df_fc = pd.concat(df_forecast)
    
    for df in [df_past, df_full, df_fc]:
        df['clade'] = df['clade'].astype(str).str.strip()
        df['frequency'] = df.groupby('date')['frequency'].transform(lambda x: x / x.sum())

    pivot_past = df_past.pivot_table(index='date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot_past = pivot_past.sort_index()
    pivot_full = df_full.pivot_table(index='date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot_full = pivot_full.sort_index()
    pivot_fc = df_fc.pivot_table(index='date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot_fc = pivot_fc.sort_index()

    label_threshold = 0.01
    max_freqs = pivot_past.max()
    clades_to_plot = max_freqs[max_freqs > label_threshold].index
    pivot_past = pivot_past[clades_to_plot]
    pivot_full = pivot_full[clades_to_plot]
    pivot_fc = pivot_fc[clades_to_plot]

    box_top = 0.92
    box_bot = 0.32
    box_lft = 0.60
    box_rt  = 0.97
    box_dy  = 0.09
    box_dx  = 0.09
    box_x  = (box_rt - box_lft - box_dx) / 2

    past_len = df_past['date'].max() - df_past['date'].min()
    fc_min = df_past['date'].max()
    fc_max = df_fc['date'].max()
    fc_len = fc_max - fc_min

    total_dy_past = box_top - box_bot
    total_dy_fc = 3 * (fc_len / past_len) * total_dy_past + 2 * box_dy
    if total_dy_fc > 1.0:
        print('Need to resize past box or reduce spacing to fit forecast boxes: total_dy_fc = %.2f' % total_dy_fc)

    fc_dy = total_dy_past * (fc_len / past_len)

    box_clade_past = dict(left=box_lft,              right=box_lft+box_x, top=box_top,                    bottom=box_bot)
    # box_clade_fc   = dict(left=box_lft+box_x+box_dx, right=box_rt,        top=box_bot+(3*fc_dy+2*box_dy), bottom=box_bot+(2*fc_dy+2*box_dy))
    # box_clade_true = dict(left=box_lft+box_x+box_dx, right=box_rt,        top=box_bot+(2*fc_dy+1*box_dy), bottom=box_bot+(1*fc_dy+1*box_dy))
    # box_clade_null = dict(left=box_lft+box_x+box_dx, right=box_rt,        top=box_bot+(1*fc_dy+0*box_dy), bottom=box_bot+(0*fc_dy+0*box_dy))
    box_clade_fc   = dict(left=box_lft+box_x+box_dx, right=box_rt,        top=box_top-(0*fc_dy+0*box_dy), bottom=box_top-(1*fc_dy+0*box_dy))
    box_clade_true = dict(left=box_lft+box_x+box_dx, right=box_rt,        top=box_top-(1*fc_dy+1*box_dy), bottom=box_top-(2*fc_dy+1*box_dy))
    box_clade_null = dict(left=box_lft+box_x+box_dx, right=box_rt,        top=box_top-(2*fc_dy+2*box_dy), bottom=box_top-(3*fc_dy+2*box_dy))

    gs_clade_past = gridspec.GridSpec(1, 1, **box_clade_past)
    ax_clade_past = plt.subplot(gs_clade_past[0, 0])
    gs_clade_fc   = gridspec.GridSpec(1, 1, **box_clade_fc)
    ax_clade_fc   = plt.subplot(gs_clade_fc[0, 0])
    gs_clade_true = gridspec.GridSpec(1, 1, **box_clade_true)
    ax_clade_true = plt.subplot(gs_clade_true[0, 0])
    gs_clade_null = gridspec.GridSpec(1, 1, **box_clade_null)
    ax_clade_null = plt.subplot(gs_clade_null[0, 0])
    
    ## past clade frequencies

    x = pivot_past.index
    Y = pivot_past.values.T
    clades = pivot_past.columns

    # colors = sns.husl_palette(len(clades))
    # colors = [k for k in sns.color_palette('Blues', n_colors=2)] + [C_BEN] + [k for k in sns.color_palette('Greens', n_colors=3)]
    colors = sns.color_palette("ch:s=-.5,r=.6", n_colors=len(clades))
    # colors = sns.cubehelix_palette(n_colors=len(clades))
    # colors = ['#3B76AF', '#B3C7E5', '#EF8636', '#BCBD46', '#DBDB96', '#ABD8E3']

    xticks      = np.linspace(np.min(df_past['date']), np.max(df_past['date'])+1, num=5)
    xlim        = [np.min(xticks), np.max(xticks)][::-1]
    # xticks[-1]  = xticks[-1] - 50 # shift slightly to avoid text overlap
    # xticks[-2]  = xticks[-2] - 33.33
    # xticks[-3]  = xticks[-3] - 16.67
    xticks      = xticks[:-1] # remove last tick to avoid overlap with text
    dx_tick     = (xticks[1] - xticks[0]) / 2
    xticks      = xticks + dx_tick # shift ticks to avoid overlap
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    xticklabels = ['\n'.join(i.strftime('%b %Y').split()) for i in xticklabels]

    # fillprops = dict(lw=AXWIDTH, ec='w')
    fillprops = dict(lw=0, ec='None')

    pprops = { 'ylim':        xlim,
               'yticks':      xticks,
               'yticklabels': xticklabels,
               'xlim':        [0, 1],
               'xticks':      [0, 0.5, 1],
               'xticklabels': ['', '', ''],
               'xminorticks': [0.25, 0.75],
               'xlabel':      'Clade frequency',
               'axoffset':    0,
               'theme':       'boxed' }
    
    cumsum = np.zeros(len(x))
    for k in range(len(clades)):
        ax_clade_past.fill_betweenx(x, cumsum, cumsum + Y[k], fc=colors[k], **fillprops)
        cumsum += Y[k]

    mp.plot(type='line', ax=ax_clade_past, x=[[-1, -1]], y=[[-1, -1]], plotprops=DEF_LINEPROPS, **pprops)

    txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'center', 'va': 'center', 'clip_on': False}
    ax_clade_past.text((box_clade_past['left']+box_clade_past['right'])/2, box_clade_past['top']+0.03, 'Past evolution', color=BKCOLOR, transform=fig.transFigure, **txtprops)
    
    ## forecast clade frequencies

    df_last = df_past[df_past['date'] == df_past['date'].max()]
    df_fc = pd.concat([df_last, df_fc])

    pivot_fc = df_fc.pivot_table(index='date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot_fc = pivot_fc.sort_index()
    pivot_fc = pivot_fc[clades_to_plot]
    x = pivot_fc.index
    Y = pivot_fc.values.T

    xticks      = np.linspace(np.min(df_fc['date']), np.max(df_fc['date'])+1, num=5)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    xticklabels = ['', '3m', '6m', '9m', '12m']

    pprops = { 'ylim':        [np.min(xticks), np.max(xticks)][::-1],
               'yticks':      xticks,
               'yticklabels': xticklabels,
               'xlim':        [0, 1],
               'xticks':      [0, 0.5, 1],
               'xticklabels': ['', '', ''],
               'xminorticks': [0.25, 0.75],
            #    'xlabel':      'Clade frequency',
               'axoffset':    0,
               'theme':       'boxed' }

    cumsum = np.zeros(len(x))
    for k in range(len(clades)):
        ax_clade_fc.fill_betweenx(x, cumsum, cumsum + Y[k], fc=colors[k], **fillprops)
        cumsum += Y[k]

    mp.plot(type='line', ax=ax_clade_fc, x=[[-1, -1]], y=[[-1, -1]], plotprops=DEF_LINEPROPS, **pprops)

    txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'center', 'va': 'center', 'clip_on': False}
    ax_clade_fc.text((box_clade_fc['left']+box_clade_fc['right'])/2, box_clade_fc['top']+0.03, 'Model forecast', color=BKCOLOR, transform=fig.transFigure, **txtprops)
    
    ## true clade frequencies

    pivot_true = df_full.pivot_table(index='date', columns='clade', values='frequency', aggfunc='sum').fillna(0)
    pivot_true = pivot_true.sort_index()
    pivot_true = pivot_true[clades_to_plot]
    x = pivot_true.index
    Y = pivot_true.values.T

    xticks      = np.linspace(np.min(df_fc['date']), np.max(df_fc['date'])+1, num=5)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    xticklabels = ['', '3m', '6m', '9m', '12m']

    pprops = { 'ylim':        [np.min(xticks), np.max(xticks)][::-1],
               'yticks':      xticks,
               'yticklabels': xticklabels,
               'xlim':        [0, 1],
               'xticks':      [0, 0.5, 1],
               'xticklabels': ['', '', ''],
               'xminorticks': [0.25, 0.75],
               'axoffset':    0,
               'theme':       'boxed' }

    cumsum = np.zeros(len(x))
    for k in range(len(clades)):
        ax_clade_true.fill_betweenx(x, cumsum, cumsum + Y[k], fc=colors[k], **fillprops)
        cumsum += Y[k]

    mp.plot(type='line', ax=ax_clade_true, x=[[-1, -1]], y=[[-1, -1]], plotprops=DEF_LINEPROPS, **pprops)

    txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'center', 'va': 'center', 'clip_on': False}
    ax_clade_true.text((box_clade_true['left']+box_clade_true['right'])/2, box_clade_true['top']+0.03, 'True frequencies', color=BKCOLOR, transform=fig.transFigure, **txtprops)
    
    ## null clade frequencies -- same as last observed in df_past

    x = df_fc['date'].unique()
    x.sort()
    Y = np.array([df_last[df_last['clade'] == clade]['frequency'].values[0] for clade in clades_to_plot])
    Y = np.tile(Y, (len(x), 1)).T

    xticks      = np.linspace(np.min(df_fc['date']), np.max(df_fc['date'])+1, num=5)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    xticklabels = ['', '3m', '6m', '9m', '12m']

    pprops = { 'ylim':        [np.min(xticks), np.max(xticks)][::-1],
               'yticks':      xticks,
               'yticklabels': xticklabels,
               'xlim':        [0, 1],
               'xticks':      [0, 0.5, 1],
               'xticklabels': ['', '', ''],
               'xminorticks': [0.25, 0.75],
               'xlabel':      'Clade frequency',
               'axoffset':    0,
               'theme':       'boxed' }

    cumsum = np.zeros(len(x))
    for k in range(len(clades)):
        ax_clade_null.fill_betweenx(x, cumsum, cumsum + Y[k], fc=colors[k], **fillprops)
        cumsum += Y[k]

    mp.plot(type='line', ax=ax_clade_null, x=[[-1, -1]], y=[[-1, -1]], plotprops=DEF_LINEPROPS, **pprops)

    txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'center', 'va': 'center', 'clip_on': False}
    ax_clade_null.text((box_clade_null['left']+box_clade_null['right'])/2, box_clade_null['top']+0.03, 'Neutral forecast', color=BKCOLOR, transform=fig.transFigure, **txtprops)

    # legend

    ax_ds[0].text(box_ds['left']-0.08, box_ds['top']+0.035, 'a'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_dw_bg[0].text(box_dw_bg['left']-0.08, box_dw_bg['top']+0.02, 'b'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_dw_rate[0].text(box_dw_rate['left']-0.08, box_dw_rate['top']+0.02, 'c'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_clade_past.text(box_clade_past['left']-0.05, box_ds['top']+0.035, 'd'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    
    # ax_ds[0].text(box_ds['left']-0.08, box_ds['top']+0.05, 'a'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    # ax_clade_past.text(box_clade_past['left']-0.08, box_ds['top']+0.05, 'b'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    # ax_clade_fc.text(box_clade_fc['left']-0.05, box_ds['top']+0.05, 'c'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    # ax_clade_true.text(box_clade_true['left']-0.05, box_clade_true['top']+0.025, 'd'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    # ax_clade_null.text(box_clade_null['left']-0.05, box_clade_null['top']+0.025, 'e'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    legend_x = box_clade_past['left'] - 0.00
    legend_y = box_clade_past['bottom'] - 0.12
    legend_dy = 0.04
    legend_dx = 0.08

    # ax_clade_past.text(legend_x, legend_y + legend_dy, 'Clades:', ha='left', va='center', color=BKCOLOR, fontsize=SIZELABEL, transform=fig.transFigure, clip_on=False)
    for idx, clade in enumerate(clades[:4]):
        clade_color = colors[idx]
        ax_clade_past.text(legend_x, legend_y - idx * legend_dy, clade, ha='left', va='center', color=clade_color, fontweight='bold', fontsize=SIZELABEL, transform=fig.transFigure, clip_on=False)

    for idx, clade in enumerate(clades[4:8]):
        clade_color = colors[idx + 4]
        ax_clade_past.text(legend_x + legend_dx, legend_y - idx * legend_dy, clade, ha='left', va='center', color=clade_color, fontweight='bold', fontsize=SIZELABEL, transform=fig.transFigure, clip_on=False)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_dw_rate(dw_files, t_min=0, dt_min=0, colors=None, fig_name='fig-dw-rate-distribution.pdf'):
    """ Plot change in background fitness distribution. """

    # Read in data
    df = [pd.read_csv(dw_file, comment='#', memory_map=True) for dw_file in dw_files]

    # Clip mutations that are too early or too short-lived
    for i in range(len(df)):
        if t_min>0 and 'T0' in df[i]:
            df[i] = df[i][df[i]['T0']>t_min]
        if dt_min>0 and 'dT' in df[i]:
            df[i] = df[i][(df[i]['T1']-df[i]['T0'])>dt_min]

    # PLOT FIGURE
    
    ## set up figure grid
    w     = ONE_FIVE_COLUMN
    goldh = w
    fig   = plt.figure(figsize=(w, goldh))

    box = dict(left=0.12, right=0.95, top=0.95, bottom=0.12)
    gs  = gridspec.GridSpec(2, 2, **box)
    ax  = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])]

    ## dw distribution

    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.7, edgecolor='none', orientation='vertical')
    labels = ['H3', 'N2', 'H1', 'N1']

    for i in range(len(df)):
        df_dw = df[i]

        # Get unique mutation names
        name_vec = []
        for it, row in df_dw.iterrows():
            name_vec.append(str(row['Position'])+'-'+str(row['Nucleotide']))
        df_dw['name'] = name_vec
        names = np.unique(df_dw['name'])

        # Get dw for fixed and lost mutations
        dw_fixed = []
        dw_lost = []
        dt_fixed = []
        dt_lost = []
        for name in names:
            row_name = df_dw[df_dw['name']==name].iloc[0]
            dt = row_name['T1'] - row_name['T0']
            if dt<dt_min or row_name['T0']<t_min:
                continue
            if row_name['Fixation']=='fixed':
                dw_fixed.append(row_name['w1_bg'] - row_name['w0_bg'])
                dt_fixed.append(dt)
            elif row_name['Fixation']=='lost':
                dw_lost.append(row_name['w1_bg'] - row_name['w0_bg'])
                dt_lost.append(dt)

        dw_fixed = np.array(dw_fixed)
        dw_lost = np.array(dw_lost)
        dt_fixed = np.array(dt_fixed)
        dt_lost = np.array(dt_lost)

        dw_fixed *= 365/dt_fixed
        dw_lost  *= 365/dt_lost

        pprops = { 'xlim'        : [-0.02, 0.025],
                   'xticks'      : [-0.02, -0.01, 0, 0.01, 0.02],
                   'xticklabels' : [-2, -1, 0, 1, 2],
                   'xminorticks' : [-0.015, -0.005, 0.005, 0.015, 0.025],
                   'ylim'        : [0, 0.40],
                   'yticks'      : [0, 0.1, 0.2, 0.3, 0.4],
                   'yminorticks' : [0.05, 0.15, 0.25, 0.35],
                #    'ylabel'      : 'Frequency',
                #    'xlabel'      : 'Change in background fitness per year (%)',
                   'bins'        : np.arange(-0.02, 0.025+0.00075, 0.00075),
                   'combine'     : True,
                   'plotprops'   : histprops,
                   'axoffset'    : 0.1,
                   'theme'       : 'open' }

        if i in [0, 1]:
            pprops['xticklabels'] = ['' for k in pprops['xticks']]
        if i in [1, 3]:
            pprops['yticklabels'] = ['' for k in pprops['yticks']]

        c_plot = [C_MPL, C_MPL] if colors is None else [colors[i], C_NEU]
        mp.hist(             ax=ax[i], x=[dw_lost],  colors=[c_plot[1]], **pprops)
        mp.plot(type='hist', ax=ax[i], x=[dw_fixed], colors=[c_plot[0]], **pprops)

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'left', 'va': 'center', 'clip_on': False}
        ax[i].text(-0.018, 0.38, '%s' % labels[i], color=BKCOLOR, **txtprops)

        t, p = st.ttest_ind(dw_fixed, dw_lost, equal_var=False)
        ax[i].text(-0.018, 0.35, 'p=%.1e' % p, color=BKCOLOR, **txtprops)

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'fontweight': 'bold', 'ha': 'right', 'va': 'center', 'clip_on': False}
        ax[i].text(0.023, 0.38, 'fixed', color=colors[i], **txtprops)
        ax[i].text(0.023, 0.35, 'lost', color=C_NEU, **txtprops)

    ax[0].text(box['left']-0.08, (box['top']+box['bottom'])/2, 'Frequency', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)
    ax[2].text((box['left']+box['right'])/2, box['bottom']-0.08, 'Change in background fitness per year (%)', transform=fig.transFigure, ha='center', va='center', **DEF_LABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_dw_bg(dw_files, t_min=0, dt_min=0, colors=None, fig_name='fig-dw-bg.pdf'):
    """ Plot change in initial background fitness. """

    # Read in data
    df = [pd.read_csv(dw_file, comment='#', memory_map=True) for dw_file in dw_files]

    # Clip mutations that are too early or too short-lived
    for i in range(len(df)):
        if t_min>0 and 'T0' in df[i]:
            df[i] = df[i][df[i]['T0']>t_min]
        if dt_min>0 and 'dT' in df[i]:
            df[i] = df[i][(df[i]['T1']-df[i]['T0'])>dt_min]

    # PLOT FIGURE
    
    ## set up figure grid
    w     = ONE_FIVE_COLUMN
    goldh = w
    fig   = plt.figure(figsize=(w, goldh))

    box = dict(left=0.12, right=0.95, top=0.95, bottom=0.12)
    gs  = gridspec.GridSpec(2, 2, **box)
    ax  = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])]

    ## dH distribution

    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.7, edgecolor='none', orientation='vertical')
    labels = ['H3', 'N2', 'H1', 'N1']

    for i in range(len(df)):
        df_dw = df[i]

        # Get unique mutation names
        name_vec = []
        for it, row in df_dw.iterrows():
            name_vec.append(str(row['Position'])+'-'+str(row['Nucleotide']))
        df_dw['name'] = name_vec
        names = np.unique(df_dw['name'])

        # Get dw for fixed and lost mutations
        dw_fixed = []
        dw_lost = []
        dt_fixed = []
        dt_lost = []
        for name in names:
            row_name = df_dw[df_dw['name']==name].iloc[0]
            dt = row_name['T1'] - row_name['T0']
            if dt<dt_min or row_name['T0']<t_min:
                continue
            if row_name['Fixation']=='fixed':
                dw_fixed.append(row_name['w0_bg']-row_name['w0_pop'])
                dt_fixed.append(dt)
            elif row_name['Fixation']=='lost':
                dw_lost.append(row_name['w0_bg']-row_name['w0_pop'])
                dt_lost.append(dt)

        dw_fixed = np.array(dw_fixed)
        dw_lost = np.array(dw_lost)
        dt_fixed = np.array(dt_fixed)
        dt_lost = np.array(dt_lost)

        # dw_fixed *= 365/dt_fixed
        # dw_lost  *= 365/dt_lost

        # print('%s\t[%.2e, %.2e]\t[%.2e, %.2e]' % (labels[i], np.min(dw_fixed), np.max(dw_fixed), np.min(dw_lost), np.max(dw_lost)))

        pprops = { 'xlim'        : [-0.02, 0.025],
                   'xticks'      : [-0.02, -0.01, 0, 0.01, 0.02],
                   'xticklabels' : [-2, -1, 0, 1, 2],
                   'xminorticks' : [-0.015, -0.005, 0.005, 0.015, 0.025],
                   'ylim'        : [0, 0.40],
                   'yticks'      : [0, 0.1, 0.2, 0.3, 0.4],
                   'yminorticks' : [0.05, 0.15, 0.25, 0.35],
                #    'ylabel'      : 'Frequency',
                #    'xlabel'      : 'Change in background fitness per year (%)',
                   'bins'        : np.arange(-0.02, 0.025+0.00075, 0.00075),
                   'combine'     : True,
                   'plotprops'   : histprops,
                   'axoffset'    : 0.1,
                   'theme'       : 'open' }

        if i in [0, 1]:
            pprops['xticklabels'] = ['' for k in pprops['xticks']]
        if i in [1, 3]:
            pprops['yticklabels'] = ['' for k in pprops['yticks']]

        c_plot = [C_MPL, C_MPL] if colors is None else [colors[i], C_NEU]
        mp.hist(             ax=ax[i], x=[dw_lost],  colors=[c_plot[1]], **pprops)
        mp.plot(type='hist', ax=ax[i], x=[dw_fixed], colors=[c_plot[0]], **pprops)

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'left', 'va': 'center', 'clip_on': False}
        ax[i].text(-0.018, 0.38, '%s' % labels[i], color=BKCOLOR, **txtprops)

        t, p = st.ttest_ind(dw_fixed, dw_lost, equal_var=False)
        ax[i].text(-0.018, 0.35, 'p=%.1e' % p, color=BKCOLOR, **txtprops)

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'fontweight': 'bold', 'ha': 'right', 'va': 'center', 'clip_on': False}
        ax[i].text(0.023, 0.38, 'fixed', color=colors[i], **txtprops)
        ax[i].text(0.023, 0.35, 'lost', color=C_NEU, **txtprops)

    ax[0].text(box['left']-0.08, (box['top']+box['bottom'])/2, 'Frequency', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)
    ax[2].text((box['left']+box['right'])/2, box['bottom']-0.08, 'Difference between initial background fitness and population average (%)', transform=fig.transFigure, ha='center', va='center', **DEF_LABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_ds(dw_files, t_min=0, dt_min=0, colors=None, fig_name='fig-ds.pdf'):
    """ Plot distribution of selection coefficients for mutations that fix versus those that are lost. """

    # Read in data
    df = [pd.read_csv(dw_file, comment='#', memory_map=True) for dw_file in dw_files]

    # Clip mutations that are too early or too short-lived
    for i in range(len(df)):
        if t_min>0 and 'T0' in df[i]:
            df[i] = df[i][df[i]['T0']>t_min]
        if dt_min>0 and 'dT' in df[i]:
            df[i] = df[i][(df[i]['T1']-df[i]['T0'])>dt_min]

    # PLOT FIGURE
    
    ## set up figure grid
    w     = ONE_FIVE_COLUMN
    goldh = w
    fig   = plt.figure(figsize=(w, goldh))

    box = dict(left=0.12, right=0.95, top=0.95, bottom=0.12)
    gs  = gridspec.GridSpec(2, 2, **box)
    ax  = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])]

    ## ds distribution

    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.7, edgecolor='none', orientation='vertical')
    labels = ['H3', 'N2', 'H1', 'N1']

    for i in range(len(df)):
        df_dw = df[i]

        # Get unique mutation names
        name_vec = []
        for it, row in df_dw.iterrows():
            name_vec.append(str(row['Position'])+'-'+str(row['Nucleotide']))
        df_dw['name'] = name_vec
        names = np.unique(df_dw['name'])

        # Get dw for fixed and lost mutations
        dw_fixed = []
        dw_lost = []
        dt_fixed = []
        dt_lost = []
        for name in names:
            row_name = df_dw[df_dw['name']==name].iloc[0]
            dt = row_name['T1'] - row_name['T0']
            if dt<dt_min or row_name['T0']<t_min:
                continue
            if row_name['Fixation']=='fixed':
                dw_fixed.append(row_name['fitness'])
                dt_fixed.append(dt)
            elif row_name['Fixation']=='lost':
                dw_lost.append(row_name['fitness'])
                dt_lost.append(dt)

        dw_fixed = np.array(dw_fixed)
        dw_lost = np.array(dw_lost)
        dt_fixed = np.array(dt_fixed)
        dt_lost = np.array(dt_lost)

        # s_vals = df[i]['fitness'].values
        # idx_max = np.argmax(s_vals)
        # print(labels[i], df[i].iloc[idx_max])

        # dw_fixed *= 365/dt_fixed
        # dw_lost  *= 365/dt_lost

        # print('%s\t[%.2e, %.2e]\t[%.2e, %.2e]' % (labels[i], np.min(dw_fixed), np.max(dw_fixed), np.min(dw_lost), np.max(dw_lost)))

        pprops = { 'xlim'        : [-0.02, 0.025],
                   'xticks'      : [-0.02, -0.01, 0, 0.01, 0.02],
                   'xticklabels' : [-2, -1, 0, 1, 2],
                   'xminorticks' : [-0.015, -0.005, 0.005, 0.015, 0.025],
                   'ylim'        : [0, 0.40],
                   'yticks'      : [0, 0.1, 0.2, 0.3, 0.4],
                   'yminorticks' : [0.05, 0.15, 0.25, 0.35],
                #    'ylabel'      : 'Frequency',
                #    'xlabel'      : 'Change in background fitness per year (%)',
                   'bins'        : np.arange(-0.02, 0.025+0.00075, 0.00075),
                   'combine'     : True,
                   'plotprops'   : histprops,
                   'axoffset'    : 0.1,
                   'theme'       : 'open' }

        if i in [0, 1]:
            pprops['xticklabels'] = ['' for k in pprops['xticks']]
        if i in [1, 3]:
            pprops['yticklabels'] = ['' for k in pprops['yticks']]

        c_plot = [C_MPL, C_MPL] if colors is None else [colors[i], C_NEU]
        mp.hist(             ax=ax[i], x=[dw_lost],  colors=[c_plot[1]], **pprops)
        mp.plot(type='hist', ax=ax[i], x=[dw_fixed], colors=[c_plot[0]], **pprops)

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'ha': 'left', 'va': 'center', 'clip_on': False}
        ax[i].text(-0.018, 0.38, '%s' % labels[i], color=BKCOLOR, **txtprops)

        t, p = st.ttest_ind(dw_fixed, dw_lost, equal_var=False)
        ax[i].text(-0.018, 0.35, 'p=%.1e' % p, color=BKCOLOR, **txtprops)

        txtprops = {'family': FONTFAMILY, 'size': SIZELABEL, 'fontweight': 'bold', 'ha': 'right', 'va': 'center', 'clip_on': False}
        ax[i].text(0.023, 0.38, 'fixed', color=colors[i], **txtprops)
        ax[i].text(0.023, 0.35, 'lost', color=C_NEU, **txtprops)

    ax[0].text(box['left']-0.08, (box['top']+box['bottom'])/2, 'Frequency', transform=fig.transFigure, rotation=90, ha='center', va='center', **DEF_LABELPROPS)
    ax[2].text((box['left']+box['right'])/2, box['bottom']-0.08, 'Inferred selection coefficient (%)', transform=fig.transFigure, ha='center', va='center', **DEF_LABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


# ___ PLOTS FOR SLIDES ETC


def plot_dH_distribution_slides(dH_file, color=None, fig_name='fig-dH-distribution-slides.pdf', t_min=0, dt_min=0, plot_rate=False):
    """ Plot EMD distribution. """

    # Read in data
    df = pd.read_csv(dH_file, comment='#', memory_map=True)

    # Clip mutations that are too early or too short-lived
    if t_min>0 and 'T0' in df:
        df = df[df['T0']>t_min]
    if dt_min>0 and 'dT' in df:
        df = df[(df['T1']-df['T0'])>dt_min]

    # Get unique mutation names
    name_vec = []
    for it, row in df.iterrows():
        name_vec.append(str(row['Position'])+'-'+str(row['Nucleotide']))
    df['name'] = name_vec
    names = np.unique(df['name'])

    # Get dH for fixed and lost mutations
    dH_fixed = []
    dH_lost = []
    dt_fixed = []
    dt_lost = []
    for name in names:
        row_name = df[df['name']==name].iloc[0]
        dt = row_name['T1'] - row_name['T0']
        if dt<dt_min or row_name['T0']<t_min:
            continue
        if row_name['Fixation']=='fixed':
            dH_fixed.append(row_name['dH'])
            dt_fixed.append(dt)
        elif row_name['Fixation']=='lost':
            dH_lost.append(row_name['dH'])
            dt_lost.append(dt)

    dH_fixed = np.array(dH_fixed)
    dH_lost = np.array(dH_lost)
    dt_fixed = np.array(dt_fixed)
    dt_lost = np.array(dt_lost)

    # PLOT FIGURE
    
    ## set up figure grid
    w     = 5.0
    goldh = 3.0
    fig   = plt.figure(figsize=(w, goldh))

    box_top = 0.96
    box_bot = 0.22
    box = dict(left=0.16, right=0.95, top=box_top, bottom=box_bot)
    gs  = gridspec.GridSpec(1, 1, **box)
    ax  = plt.subplot(gs[0, 0])

    # b -- dH distribution

    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.7, edgecolor='none', orientation='vertical')

    pprops = { 'xlim'        : [-0.02, 0.04],
               'xticks'      : [-0.02, 0, 0.02, 0.04],
               'xticklabels' : [-2, 0, 2, 4],
               'xminorticks' : [-0.01, 0.01, 0.03],
            #    'ylim'        : [0.01, 1],
            #    'yticks'      : [0.01, 0.1, 1],
               'ylim'        : [0, 0.40],
               'yticks'      : [0, 0.1, 0.2, 0.3, 0.4],
               'yminorticks' : [0.05, 0.15, 0.25, 0.35],
               'ylabel'      : 'Frequency',
               'xlabel'      : 'Change in background fitness (%)',
               'bins'        : np.arange(-0.02, 0.04+0.001, 0.001),
               'combine'     : True,
               'plotprops'   : histprops,
               'axoffset'    : 0.1,
               'theme'       : 'open' }
    
    # ax.set_yscale('log')

    if plot_rate:
        dH_fixed *= 365/dt_fixed
        dH_lost  *= 365/dt_lost

        pprops['xlim']   = [-0.02, 0.025]
        pprops['xticks'] = [-0.02, -0.01, 0, 0.01, 0.02]
        pprops['xticklabels'] = [-2, -1, 0, 1, 2]
        pprops['xminorticks'] = [-0.015, -0.005, 0.005, 0.015, 0.025]
        pprops['bins'] = np.arange(-0.02, 0.025+0.00075, 0.00075)
        pprops['xlabel'] = 'Change in background fitness / year (%)'

    print(np.min(dH_fixed), np.max(dH_fixed))
    print(np.min(dH_lost), np.max(dH_lost))

    c_plot = [C_MPL, C_MPL] if color is None else [color, C_NEU]
    mp.hist(             ax=ax, x=[dH_lost],  colors=[c_plot[1]], **pprops)
    mp.plot(type='hist', ax=ax, x=[dH_fixed], colors=[c_plot[0]], **pprops)
    # mp.plot(type='hist', ax=ax, x=[dH_lost],  colors=[c_plot[1]], **pprops)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_figure_HA_circle(name, HA_file, landmark_start=[1], landmark_end=[566], landmark_label=['HA']):
    """
    HA inferred selection coefficients.
    """

    # PLOT FIGURE

    ## set up figure grid

    w       = SINGLE_COLUMN # SLIDE_WIDTH
    hshrink = 1
    goldh   = w * hshrink
    fig     = plt.figure(figsize=(w, goldh))

    box_top = 0.97
    box_bot = 0.03
    dy      = 0 #0.08  
    y_circ  = (box_top - box_bot - dy) * (5/5)  # decide the relative height of the circle plots
    y_enr   = (box_top - box_bot - dy) * (0/5)  # decide the relative height of the enrichment plots
    x_circ  = y_circ * goldh / w  # ensure axes are square
    dx      = (1 - x_circ) / 3  # position circle plots in center
    x0      = -0.02

    box_circ = dict(left=x0 + 1*dx + 0*x_circ, right=x0 + 1*dx + 1*x_circ, bottom=box_top-y_circ, top=box_top)
    gs_circ  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_circ)
    ax_circ  = plt.subplot(gs_circ[0, 0])
    
    #dx = -0.08
    dy_sub   = 0.02
    dy_label = 0.035
    dx_label = 0.033

    ## a -- circle plot

#    # _ SLIDES
#    w       = 8
#    fig     = plt.figure(figsize=(w, w))
#    ax_circ = plt.subplot(111)
#    # ^ SLIDES

    label2ddr = { 'tat exon 2': 0.13,
                  'rev exon 2': 0.10 }

    # HA_s     = 'data/selection_coeff_H3N2_HA_since_01_2009.csv'
    # HA_index = 'data/703010505-3-index.csv'

    sig_s, sig_site_real = plot_circle(ax_circ, HA_file, landmark_start, landmark_end, landmark_label, label2ddr)

    ax_circ.text(box_circ['left']+dy_sub, box_circ['top']+dy_sub, 'a'.upper(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_circ.text(box_circ['left']+dx_label, box_circ['top']-dy_label, '%s HA' % name, transform=fig.transFigure, **DEF_LABELPROPS)

#    # _ SLIDES
#    plt.savefig('%s/new-slides-cap256-circle.pdf' % FIG_DIR, dpi=1000, **FIGPROPS)
#    plt.savefig('%s/new-slides-cap256-circle.png' % FIG_DIR, dpi=1000, **FIGPROPS)
#    plt.close(fig)
#    w       = 8
#    fig     = plt.figure(figsize=(w, 2.85))
#    ax_smpl = plt.subplot(311)
#    # ^ SLIDES

#     dx =  0.04
#     dy = -0.02
#     ax_circ.text(0.06, box_circ['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

#     ## a -- trajectory plot

#     pprops = { 'xticks':      [30, 190, 350, 510, 670],
#                'yticks':      [0, 1],
#                'yminorticks': [0.25, 0.5, 0.75],
#                'nudgey':      1.1,
#                'xlabel':      'Time (days)',
#                'ylabel':      'Variant frequency\nin VRC26 epitope',
#                'plotprops':   {'lw': 1.5*SIZELINE, 'ls': '-', 'alpha': 0.75 },
#                'axoffset':    0.1,
#                'theme':       'open' }

#     for i in range(len(var_tag)-1):
#         xdat = [times]
#         ydat = [var_traj[i]]
#         mp.line(ax=ax_traj, x=xdat, y=ydat, colors=[var_c[i]], **pprops)

#     xdat = [times]
#     ydat = [var_traj[len(var_tag)-1]]
#     mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=[var_c[len(var_tag)-1]], **pprops)

#     ax_traj.text(0.06, box_traj['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

#     invt = ax_traj.transData.inverted()
#     xy1  = invt.transform((  0, 0))
#     xy2  = invt.transform((7.5, 9))
#     xy3  = invt.transform((3.0, 9))

#     legend_dx1 = 1*(xy1[0]-xy2[0])
#     legend_dx2 = 1*(xy1[0]-xy3[0])
#     legend_dy  = 1*(xy1[1]-xy2[1])

#     traj_legend_x =  80
#     traj_legend_y =  0.95
#     traj_legend_d = -0.05
#     traj_legend_t = ['Superinfecting\nvariant', 'Other']
#     traj_legend_c = [C_MPL, C_NEU]
#     for k in range(len(traj_legend_t)):
#         mp.line(ax=ax_traj, x=[[traj_legend_x + legend_dx1, traj_legend_x + legend_dx2]],
#                 y=[[traj_legend_y + (1.5 * k * legend_dy), traj_legend_y + (1.5 * k * legend_dy)]],
#                 colors=[traj_legend_c[k]], plotprops=dict(lw=1.5*SIZELINE, ls='-', clip_on=False))
#         ax_traj.text(traj_legend_x, traj_legend_y + (1.5 * k * legend_dy), traj_legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

#     ## c -- selection in the VRC26 epitope

#     site_rec_props  = dict(height=1, width=1, ec=None, lw=AXWIDTH/2, clip_on=False)
#     codon_rec_props = dict(height=4, width=3, ec=BKCOLOR, fc='none', lw=AXWIDTH/2, clip_on=False)
#     cBG             = '#F5F5F5'
#     rec_patches     = []
#     TF_dots_x       = [ 0.5]
#     TF_dots_y       = [-3.5]

#     sig_s         = np.array(sig_s)
#     sig_site_real = np.array(sig_site_real)
#     sig_nuc_idx   = np.array(sig_nuc_idx)
#     eidx          = 0
#     sub_box       = 0
#     sub_dx        = 3 + 0.3
#     for i in range(epitope_start[eidx], epitope_end[eidx]+1, 3):
#         for sub_i in range(3):
#             TF_dots_x.append(sub_box*sub_dx + sub_i + 0.5)
#             TF_dots_y.append(4-NUC.index(df_index.iloc[i+sub_i].TF)+0.5)
#             idxs     = sig_site_real==i+sub_i
#             temp_s   = sig_s[idxs]
#             temp_nuc = sig_nuc_idx[idxs]
#             for j in range(len(NUC)-1):
#                 if df_index.iloc[i+sub_i].TF==NUC[j+1]:
#                     continue
#                 c = cBG
#                 if j in temp_nuc:
#                     t = temp_s[list(temp_nuc).index(j)] / 0.05
#                     if np.fabs(t)>1:
#                         t /= np.fabs(t)
#                     if t>0:
#                         c = hls_to_rgb(0.02, 0.53 * t + 1. * (1 - t), 0.83)
#                     else:
#                         c = hls_to_rgb(0.58, 0.53 * np.fabs(t) + 1. * (1 - np.fabs(t)), 0.60)
#                 rec = matplotlib.patches.Rectangle(xy=(sub_box*sub_dx + sub_i, 3-j), fc=c, **site_rec_props)
#                 rec_patches.append(rec)
#         rec = matplotlib.patches.Rectangle(xy=(sub_box*sub_dx, 0), **codon_rec_props)
#         rec_patches.append(rec)
#         sub_box += 1

#     for i in range(-5, 5+1, 1):
#         c = cBG
#         t = i/5
#         if t>0:
#             c = hls_to_rgb(0.02, 0.53 * t + 1. * (1 - t), 0.83)
#         else:
#             c = hls_to_rgb(0.58, 0.53 * np.fabs(t) + 1. * (1 - np.fabs(t)), 0.60)
#         rec = matplotlib.patches.Rectangle(xy=(sub_box*sub_dx - 6.5 + i, -4), fc=c, **site_rec_props)
#         rec_patches.append(rec)

#     invt = ax_smpl.transData.inverted()
#     xy1  = invt.transform((0,0))
#     xy2  = invt.transform((0,9))
#     coef_legend_dy = (xy1[1]-xy2[1]) # multiply by 3 for slides/poster
#     c   = cBG
#     rec = matplotlib.patches.Rectangle(xy=(0, -4 + 4*coef_legend_dy), fc=c, **site_rec_props) # paper
# #    rec = matplotlib.patches.Rectangle(xy=(0, -4 + 6*coef_legend_dy), fc=c, **site_rec_props) # slides
#     rec_patches.append(rec)

#     for patch in rec_patches:
#         ax_smpl.add_artist(patch)

#     pprops = { 'colors': [BKCOLOR],
#                'xlim': [0, epitope_end[eidx]-epitope_start[eidx] + (sub_box * (sub_dx - 3)) + 1],
#                'ylim': [0, 4.1],
#                'xticks': [],
#                'yticks': [],
#                'plotprops': dict(lw=0, s=0.2*SMALLSIZEDOT, marker='o', clip_on=False),
#                'ylabel': '',
#                'theme': 'open',
#                'hide' : ['top', 'bottom', 'left', 'right'] }

#     mp.plot(type='scatter', ax=ax_smpl, x=[TF_dots_x], y=[TF_dots_y], **pprops)

#     ax_smpl.text(0.06, box_smpl['top']+0.02, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

#     txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
#     for i in range(len(NUC)-1):
#         ax_smpl.text(-0.85, 3-i+0.5, NUC[i+1], clip_on=False, **txtprops)

#     txtprops['ha'] = 'left'
#     ax_smpl.text(1.3, -3.5, 'TF nucleotide', clip_on=False, **txtprops)
#     ax_smpl.text(1.3, -3.5 + 4*coef_legend_dy, 'Not observed', clip_on=False, **txtprops) # paper
# #    ax_smpl.text(1.3, -3.5 + 6*coef_legend_dy, 'Not observed', clip_on=False, **txtprops) # slides

#     txtprops['ha'] = 'center'
#     txtprops['va'] = 'top'
#     for i in range((epitope_end[eidx]-epitope_start[eidx]+1)//3):
#         ax_smpl.text(1.5+i*sub_dx, -0.5, 160+i, clip_on=False, **txtprops)

#     ax_smpl.text(sub_box*sub_dx - 11, -4.5, -5, clip_on=False, **txtprops)
#     ax_smpl.text(sub_box*sub_dx -  6, -4.5,  0, clip_on=False, **txtprops)
#     ax_smpl.text(sub_box*sub_dx -  1, -4.5,  5, clip_on=False, **txtprops)
#     ax_smpl.text(sub_box*sub_dx -  5.5, -6.0, 'Inferred selection\ncoefficient, $\hat{s}$ (%)', clip_on=False, **txtprops)

# #    # _ SLIDES
# #    plt.savefig('%s/new-slides-cap256-selection.pdf' % FIG_DIR, dpi=1000, **FIGPROPS)
# #    plt.savefig('%s/new-slides-cap256-selection.png' % FIG_DIR, dpi=1000, **FIGPROPS)
# #    plt.close(fig)
# #    # ^ SLIDES

    # MAKE LEGEND

    invt = ax_circ.transData.inverted()
    xy1  = invt.transform((0,0))
    xy2  = invt.transform((0,15))

    coef_legend_x  = 0.90
    coef_legend_dx = 0.05
    coef_legend_y  = -0.75
    coef_legend_dy = xy1[1]-xy2[1]

    txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
    s_mult   = 40*SMALLSIZEDOT
    ex_s     = [-0.015, -0.012, -0.009, -0.006, -0.003, 0, 0.003, 0.006, 0.009, 0.012, 0.015]
    show_s   = [   1,     0,     0,     0,     0, 1,    0,    0,    0,    0,    1]
    c_s      = [C_DEL, C_BEN]
    for i in range(len(ex_s)):
        plotprops = dict(lw=0, marker='o', s=np.fabs(ex_s[i])*s_mult, clip_on=False)
        mp.scatter(ax=ax_circ, x=[[coef_legend_x + i*coef_legend_dx]], y=[[coef_legend_y]], colors=[c_s[ex_s[i]>0]], plotprops=plotprops)
        if show_s[i]:
            ax_circ.text(coef_legend_x + i*coef_legend_dx, coef_legend_y + 0.75*coef_legend_dy, '%d' % (100*ex_s[i]), **txtprops)
    ax_circ.text(coef_legend_x + 5*coef_legend_dx, coef_legend_y + (2.25*coef_legend_dy), 'Inferred selection\ncoefficient, ' + r'$\hat{s}$' + ' (%)', **txtprops)

    # SAVE FIGURE

    plt.savefig('fig-%s-HA-circle%s' % (name, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)

    print('%s HA done.' % name)


def plot_circle(ax, s_file, landmark_start, landmark_end, landmark_label, label2ddr):
    """
    Create a circle plot showing selection, large elements of the inverse covariance, and HIV sequence features.
    """

    df = pd.read_csv('%s' % (s_file), comment='#', memory_map=True)
    # df_index = pd.read_csv('%s' % (index_file), comment='#', memory_map=True)

    # cov  = np.loadtxt('%s/covariance-%s-poly-seq2state.dat' % (HIV_MPL_DIR, tag))
    # num  = np.loadtxt('%s/numerator-%s-poly-seq2state.dat' % (HIV_MPL_DIR, tag))
    # cinv = np.linalg.inv(cov)
    # ds   = cinv / 1e4

    # Check for selection coefficients that are significant: |s| > mult * sigma_s

    s_min = 0.001
    n_sig = 0

    sig_s         = []
    sig_site_real = []

    #print('%s\nvariant\ts\t(sigma)' % (tag))
    for df_iter, df_entry in df.iterrows():
        if df_entry['Mutation']!=df_entry['Reference']:
            s_val = df_entry['Selection_coefficient']
            if np.fabs(s_val)>s_min:
                n_sig += 1
                sig_s.append(s_val)
                sig_site_real.append(df_entry['Position'])

#    print('')
#    print('%d/%d (%d%%) significant at %.2f sigma' % (n_sig, len(df_info), 100*n_sig/len(df_info), mult))
#    print('')

    # Sequence landmarks

    # seq_range = [[ 1, 566], [2085, 5096], [5041, 5619], [5559, 5850], [5831, 6045], [5970, 6045], [6045, 6310]]
    # seq_range = seq_range +   [[6225, 8795], [8379, 8469], [8379, 8653], [8797, 9417], [9086, 9719]]
    # seq_label = [       'gag',        'pol',        'vif',        'vpr', 'tat exon 1', 'rev exon 1',        'vpu']
    # seq_label = seq_label +  [        'env', 'tat exon 2', 'rev exon 2',        'nef',     "3' LTR"]

    # landmark_start = [1]
    # landmark_end   = [566]
    # landmark_label = ['HA']

    # idx = int(df_index.iloc[0].HXB2)
    # for i in range(len(df_index)):
    #     try:
    #         idx = int(df_index.iloc[i].HXB2)
    #     except:
    #         pass
    #     for j in range(len(seq_range)):
    #         if idx==seq_range[j][0] or (i==0 and idx>seq_range[j][0] and idx<seq_range[j][1]):
    #             landmark_start.append(i)
    #             landmark_end.append(i)
    #             landmark_label.append(seq_label[j])
    #         if idx==seq_range[j][1] or (i==len(df_index)-1 and idx>seq_range[j][0] and int(df_index.iloc[0].HXB2)<=seq_range[j][0]
    #                                     and idx<seq_range[j][1]):
    #             landmark_end[landmark_label.index(seq_label[j])] = i

    # Epitope labels

    # seq_range = epitope_range.copy()
    # seq_label = epitope_label.copy()

    # epitope_start = []
    # epitope_end   = []
    # epitope_label = []
    # epitope_sites = []
    # site2epitope  = {}

    # idx = int(df_index.iloc[0].HXB2)
    # for i in range(len(df_index)):
    #     try:
    #         idx = int(df_index.iloc[i].HXB2)
    #     except:
    #         pass
    #     for j in range(len(seq_range)):
    #         if idx==seq_range[j][0] or (i==0 and idx>seq_range[j][0] and idx<seq_range[j][1]):
    #             epitope_start.append(i)
    #             epitope_end.append(i)
    #             epitope_label.append(seq_label[j])
    #             if seq_label[j]=='DG9':
    #                 epitope_start[-1] -= 9 # account for DEP insertion
    #         if idx==seq_range[j][1] or (i==len(df_index)-1 and idx>seq_range[j][0] and int(df_index.iloc[0].HXB2)<=seq_range[j][0]
    #                                     and idx<seq_range[j][1]):
    #             epitope_end[epitope_label.index(seq_label[j])] = i
    #             iix = epitope_label.index(seq_label[j])
    #             for k in range(epitope_start[iix],epitope_end[iix]):
    #                 epitope_sites.append(k)
    #                 if k in site2epitope:
    #                     print('Unexpected! Overlapping epitopes.')
    #                 else:
    #                     site2epitope[k] = seq_label[j]
    #             print(''.join(list(df_index.iloc[epitope_start[iix]:epitope_end[iix]].TF)))


    # Dot plot for significant selection coefficients

    c_dot  = { True: C_BEN, False: C_DEL }
    s_mult = 60*SMALLSIZEDOT

    start_idx = 0
    end_idx   = 566

    def_buff = 50
    def site2angle(site, deg=False, buffer=def_buff):
        if deg: return    -360.*(site+(buffer/2))/(end_idx-start_idx+buffer)
        else:   return -2*np.pi*(site+(buffer/2))/(end_idx-start_idx+buffer)

    print(np.max(sig_s), np.min(sig_s))

    # s_max    = 0.125
    # s_min    = -0.05
    # s_norm   = s_max - s_min
    # s_levels = [-0.05, -0.025, 0, 0.025, 0.05, 0.075, 0.10, 0.125]

    # Compress s range for plotting clarity
    s_max    = 0.015
    s_min    = -0.01
    s_norm   = s_max - s_min
    s_levels = [-0.01, -0.005, 0, 0.005, 0.01, 0.015]
    sig_s    = np.array(sig_s)
    sig_s[sig_s>s_max] = s_max

    r_max = 0.90
    r_min = 0.45
    s2r = lambda s: ((s-s_min)/s_norm) * (r_max-r_min) + r_min
    level_rad = [s2r(s) for s in s_levels]

    dot_colors = [c_dot[s>0]          for s in sig_s]
    dot_sizes  = [s_mult * np.fabs(s) for s in sig_s]
    dot_rads   = [s2r(s) for s in sig_s]
    dot_x = [dot_rads[i] * np.cos(site2angle(sig_site_real[i])) for i in range(len(sig_s))]
    dot_y = [dot_rads[i] * np.sin(site2angle(sig_site_real[i])) for i in range(len(sig_s))]

    txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, rotation=0)
    s_text_levels = [-0.05, 0, 0.05]
    text_level_rad = [s2r(s) for s in s_text_levels]
    for i in range(len(s_text_levels)):
        ax.text(text_level_rad[i], 0, int(100*s_text_levels[i]), **txtprops)

    # Lines for polymorphic sites

    # line_r = [0.67, 0.71]
    # line_x = [[line_r[0] * np.cos(site2angle(i)), line_r[1] * np.cos(site2angle(i))] for i in np.unique(df.alignment_index)]
    # line_y = [[line_r[0] * np.sin(site2angle(i)), line_r[1] * np.sin(site2angle(i))] for i in np.unique(df.alignment_index)]
    # line_c = [LCOLOR                                                                 for i in np.unique(df.alignment_index)]

    # Arcs for  sequence landmarks

    arc_r            = [0.96, 0.99, 1.02]
    arc_dr           = 0.01
    track            = 0
    landmark_patches = []
    landmark_text_d  = { 'pol'    : -40,
                         'vif'    :  10,
                         "3' LTR" :  40 }

    for i in range(len(landmark_label)):
        if i>0 and landmark_start[i]<landmark_end[i-1]:
            track = (track + 1)%len(arc_r)
        
        arc_props = dict(center=[0,0], r=arc_r[track], width=arc_dr, lw=AXWIDTH/2, ec=BKCOLOR, fc='none',
                         theta1=site2angle(landmark_end[i], deg=True), theta2=site2angle(landmark_start[i], deg=True))
        landmark_patches.append(matplotlib.patches.Wedge(**arc_props))

        # label with line
        if landmark_label[i] in ['Signal peptide']:
            txtprops  = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, rotation=0)
            plotprops = dict(lw=AXWIDTH/2, ls='-', clip_on=False)

            label_x = [arc_r[track]*np.cos(site2angle(landmark_start[i]))]
            label_y = [arc_r[track]*np.sin(site2angle(landmark_start[i]))]
            ddr     = 0.11
            if landmark_label[i] in label2ddr:
                ddr = label2ddr[landmark_label[i]]

            ddx1 =  ddr * np.cos(site2angle(landmark_start[i]))
            ddy  =  ddr * np.sin(site2angle(landmark_start[i]))
            ddx2 = ddx1 + np.sign(ddx1)*0.03

            label_x = label_x + [label_x[0] + ddx1, label_x[0] + ddx2]
            label_y = label_y + [label_y[0] +  ddy, label_y[0] +  ddy]
            if label_x[0]<0:
                txtprops['ha'] = 'right'
            else:
                txtprops['ha'] = 'left'

            ax.text(label_x[-1], label_y[-1], landmark_label[i], **txtprops)
            mp.line(ax=ax, x=[label_x], y=[label_y], colors=[BKCOLOR], plotprops=plotprops)

        # plot normally
        else:
            delta_site = 50
            if landmark_label[i] in ['vif']:
                delta_site += landmark_text_d[landmark_label[i]] + 25
            elif landmark_label[i]=='vpu' and '703010505' in s_file:
                delta_site = 25
            elif landmark_label[i]=='env' and '703010505' in s_file:
                delta_site = 75
            elif landmark_label[i] in ["3' LTR"] or (landmark_label[i]=='pol' and '700010077' in s_file):
                delta_site += landmark_text_d[landmark_label[i]]
            txt_dr   = 0.04
            txt_ang  = site2angle(landmark_start[i]+delta_site)
            txt_rot  = site2angle(landmark_start[i]+delta_site, deg=True)-90
            txt_x    = (arc_r[track] + (arc_dr/2) + txt_dr)*np.cos(txt_ang)
            txt_y    = (arc_r[track] + (arc_dr/2) + txt_dr)*np.sin(txt_ang)
            txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY,
                            size=SIZELABEL, rotation=txt_rot)
            ax.text(txt_x, txt_y, landmark_label[i], **txtprops)

    # Arcs for TGCA selection tracks
    
    for i in range(len(level_rad)):
        arc_props = dict(center=[0,0], r=level_rad[i], width=0, lw=AXWIDTH/2, ec=C_NEU_LT, fc='none', theta1=0, theta2=360)
        landmark_patches.append(matplotlib.patches.Wedge(**arc_props))

    # Outer arcs for additional sequence annotations
        
    # arc_range = [[6615, 6695], [6696, 6782], [7047, 7073], [7110, 7217], [7377, 7478], [7602, 7634]]
    # arc_label = [       'V1',          'V2',     'loop D',         'V3',         'V4',         'V5']

    # landmark_start = []
    # landmark_end   = []
    # landmark_label = []

    # idx = int(df_index.iloc[0].HXB2)
    # for i in range(len(df_index)):
    #     try:
    #         idx = int(df_index.iloc[i].HXB2)
    #     except:
    #         pass
    #     for j in range(len(arc_range)):
    #         if idx==arc_range[j][0] or (i==0 and idx>arc_range[j][0] and idx<arc_range[j][1]):
    #             landmark_start.append(i)
    #             landmark_end.append(i)
    #             landmark_label.append(arc_label[j])
    #         if idx==arc_range[j][1] or (i==len(df_index)-1 and idx>arc_range[j][0] and int(df_index.iloc[0].HXB2)<=arc_range[j][0]
    #                                     and idx<arc_range[j][1]):
    #             landmark_end[landmark_label.index(arc_label[j])] = i

    # arc_r = 1.05
    # for i in range(len(landmark_label)):
    #     arc_props = dict(center=[0,0], r=arc_r, width=0, lw=AXWIDTH/2, ec=BKCOLOR, fc='none',
    #                      theta1=site2angle(landmark_end[i], deg=True), theta2=site2angle(landmark_start[i], deg=True))
    #     landmark_patches.append(matplotlib.patches.Wedge(**arc_props))

    #     # plot normally
    #     delta_site = 50
    #     if landmark_label[i]=='V2' and '703010848' in s_file:
    #         delta_site = 75
    #     elif landmark_label[i]=='loop D' and '703010505' in s_file:
    #         delta_site = 0
    #     elif landmark_label[i]=='loop D' and '703010848' in s_file:
    #         delta_site = 0
    #     elif landmark_label[i]=='V3' and '703010848' in s_file:
    #         delta_site = 65
    #     elif landmark_label[i]=='V5' and '703010505' in s_file:
    #         delta_site = 15
    #     elif landmark_label[i]=='V5' and '703010848' in s_file:
    #         delta_site = 15
    #     txt_dr   = 0.04
    #     txt_ang  = site2angle(landmark_start[i]+delta_site)
    #     txt_rot  = site2angle(landmark_start[i]+delta_site, deg=True)-90
    #     txt_x    = (arc_r + txt_dr)*np.cos(txt_ang)
    #     txt_y    = (arc_r + txt_dr)*np.sin(txt_ang)
    #     txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY,
    #                     size=SIZELABEL, rotation=txt_rot)
    #     ax.text(txt_x, txt_y, landmark_label[i], **txtprops)

    # # Arcs for epitopes

    # arc_r           = 1.04
    # arc_dr          = 0.32
    # epitope_patches = []

    # for i in range(len(epitope_label)):
        
    #     arc_props = dict(center=[0,0], r=arc_r, width=arc_dr, lw=AXWIDTH/2, ec=BKCOLOR, fc='#f2f2f2', alpha=0.8,
    #                      theta1=site2angle(epitope_end[i], deg=True), theta2=site2angle(epitope_start[i], deg=True))
    #     epitope_patches.append(matplotlib.patches.Wedge(**arc_props))

    #     # label epitopes
    #     if True:
    #         mid       = (site2angle(epitope_end[i], deg=True)+site2angle(epitope_start[i], deg=True))/2.
    #         label_x   = [arc_r * np.cos(mid * 2 * np.pi / 360.)]
    #         label_y   = [arc_r * np.sin(mid * 2 * np.pi / 360.)]
    #         ddr       = 0.06
    #         if epitope_label[i] in label2ddr:
    #             ddr = label2ddr[epitope_label[i]]
    #         ddx1 =  ddr * np.cos(mid * 2 * np.pi / 360.)
    #         ddy  =  ddr * np.sin(mid * 2 * np.pi / 360.)
    #         ddx2 = ddx1 + np.sign(ddx1)*0.03
    #         txtprops  = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL, rotation=0)
    #         plotprops = dict(lw=AXWIDTH/2, ls='-', clip_on=False)

    #         label_x = label_x + [label_x[0] + ddx1, label_x[0] + ddx2]
    #         label_y = label_y + [label_y[0] +  ddy, label_y[0] +  ddy]
    #         if label_x[0]<0:
    #             txtprops['ha'] = 'right'
    #         else:
    #             txtprops['ha'] = 'left'

    #         ax.text(label_x[-1], label_y[-1], epitope_label[i], **txtprops)
    #         mp.line(ax=ax, x=[label_x], y=[label_y], colors=[BKCOLOR], plotprops=plotprops)

    # # Arc plot for large values of integrated covariance

    # c_pos  = LCOLOR
    # c_neg  = LCOLOR
    # c_circ = { True : c_neg, False : c_pos }

    # arc_rad   = 0.65
    # arc_mult  = SIZELINE/2
    # arc_alpha = 0.1

    # circ_color = [c_circ[ic>0]                                     for ic in inv_cov]
    # #circ_color = [c_circ[idx_1[i] in epitope_sites or idx_2[i] in epitope_sites] for i in range(len(inv_cov))]
    # circ_rad   = [[arc_rad, arc_rad]                               for ic in inv_cov]
    # circ_arc   = [dict(lw=arc_mult * np.fabs(ic), alpha=arc_alpha) for ic in inv_cov]
    # circ_x     = [(i+(def_buff/2)) for i in idx_1]
    # circ_y     = [(i+(def_buff/2)) for i in idx_2]

    # Make plot

    pprops = { 'xlim':   [-1.05, 1.05],
               'ylim':   [-1.05, 1.05],
               'xticks': [],
               'yticks': [],
               'noaxes': True }

    plotprops = dict(lw=0, marker='o', s=dot_sizes[0], zorder=9999)
    mp.plot(type='scatter', ax=ax, x=[[dot_x[0]]], y=[[dot_y[0]]], colors=[dot_colors[0]], plotprops=plotprops, **pprops)

    for i in range(1, len(dot_x)):
        plotprops = dict(lw=0, marker='o', s=dot_sizes[i], zorder=9999)
        mp.scatter(ax=ax, x=[[dot_x[i]]], y=[[dot_y[i]]], colors=[dot_colors[i]], plotprops=plotprops)

    # for i in range(len(np.unique(df_info.alignment_index))):
    #     plotprops = dict(lw=AXWIDTH, ls='-')
    #     mp.line(ax=ax, x=[line_x[i]], y=[line_y[i]], colors=[line_c[i]], plotprops=plotprops)

    # mp.plot(type='circos', ax=ax, x=circ_x, y=circ_y, **pprops)

    for patch in landmark_patches:
        ax.add_artist(patch)

    # for patch in epitope_patches:
    #     ax.add_artist(patch)

    return sig_s, sig_site_real


def plot_AUC_grant(auc_files, plot_null=True, plot_model=True, hide_left=False, fig_name='fig-AUC.pdf'):
    """ AUC bar plots predicting fixation of mutations. """

    # Read in data
    dfs = [pd.read_csv(f, comment='#', memory_map=True) for f in auc_files]
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = 3.35
    goldh = w * 0.60
    fig   = plt.figure(figsize=(w, goldh))

    box = dict(left=0.18, right=0.96, top=0.94, bottom=0.18)
    gs  = gridspec.GridSpec(1, 2, **box)
    axs = [plt.subplot(gs[0, i]) for i in range(2)]

    histprops = dict(lw=AXWIDTH/2, width=0.9*0.1, align='center', orientation='vertical', edgecolor=[BKCOLOR], alpha=0.5)
    
    pprops = { 'xlim':        [0, 1],
               'xticks':      [0, 0.5, 1],
               'xticklabels': ['0', '0.5', '1'],
               'xlabel':      'H3N2 mutation frequency',
               'ylim':        [0., 1],
               'yticks':      [0., 0.5, 1],
               'yminorticks': [0.25, 0.75],
               'ylabel':      'Accuracy of mutation fixation\nprediction (AUC)',
               'plotprops':   histprops,
               'theme':       'open',
               'hide':        ['top', 'right'] }

    for i in range(len(dfs)):
        if i>0:
            pprops['xlabel']      = 'H1N1 mutation frequency'
            pprops['ylabel']      = ''
            pprops['yticks']      = []
            pprops['yticklabels'] = []
            pprops['yminorticks'] = []
            pprops['hide'] = ['top', 'right', 'left']

        if plot_model:
            pprops['colors'] = [C_MPL]
            mp.bar(ax=axs[i], x=[dfs[i]['Frequency']], y=[dfs[i]['AUC_simulation']], **pprops)

        if plot_null:
            pprops['colors'] = [C_NEU]
            mp.plot(type='bar', ax=axs[i], x=[dfs[i]['Frequency']], y=[dfs[i]['AUC_null']], **pprops)

    # axs[0].text(box['left']+0.05, box['top'], 'H3N2', transform=fig.transFigure, **DEF_SUBLABELPROPS)
    # axs[0].text(box['left']+0.45, box['top'], 'H1N1', transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # # legend
    
    # colors    = [C_MPL, C_NEU]
    # colors_lt = [C_MPL, C_NEU_LT]
    # plotprops = DEF_ERRORPROPS.copy()
    # plotprops['clip_on'] = False
    
    # invt = axs[0].transData.inverted()
    # xy1  = invt.transform((   0,  0))
    # xy2  = invt.transform((7.50, 10))
    # xy4  = invt.transform((5.25, 10))

    # legend_dx = xy1[0]-xy4[0]
    # legend_dy = xy1[1]-xy2[1]
    # legend_x  = 0.65
    # legend_y  = [1.0, 1.0 + 1*legend_dy]
    # legend_t  = ['Our model', 'Neutral']
    
    # for k in range(len(dfs)):
    #     mp.error(ax=axs[0], x=[[legend_x+legend_dx]], y=[[legend_y[k]]], edgecolor=[colors[k]], facecolor=[colors_lt[k]], plotprops=plotprops)
    #     axs[0].text(legend_x, legend_y[k], legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_overview_grant(freq_file, s_files, fig_name='fig-overview.pdf'):
    """ Plot H3N2 HA trajectories, selection coefficients based on whether or not mutations are in antigenic sites. """

    # Read in data
    x = np.loadtxt(freq_file)
    above_cut = np.array([np.max(x[i])>0.01 for i in range(len(x))])
    x = x[above_cut]
    not_WT = np.array([x[i][0]<0.95 for i in range(len(x))])
    x = x[not_WT]
    t = np.arange(0, len(x[0])*10, 10)
    
    dfs = [pd.read_csv(f, comment='#', memory_map=True) for f in s_files]
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = 3.35
    goldh = 1.2*w
    fig   = plt.figure(figsize=(w, goldh))

    dy = 0.12
    box_top = 0.94
    box_bot = 0.11
    box_y = (box_top - box_bot - 2*dy) / 3
    box_traj = dict(left=0.18, right=0.96, top=box_top, bottom=box_top - box_y)
    box_sH3N2 = dict(left=0.18, right=0.96, top=box_top - 1*box_y - 1*dy, bottom=box_top - 2*box_y - 1*dy)
    box_sH1N1 = dict(left=0.18, right=0.96, top=box_top - 2*box_y - 2*dy, bottom=box_top - 3*box_y - 2*dy)

    gs_traj  = gridspec.GridSpec(1, 1, **box_traj)
    gs_sH3N2 = gridspec.GridSpec(1, 1, **box_sH3N2)
    gs_sH1N1 = gridspec.GridSpec(1, 1, **box_sH1N1)

    ax_traj  = plt.subplot(gs_traj[0, 0])
    ax_sH3N2 = plt.subplot(gs_sH3N2[0, 0])
    ax_sH1N1 = plt.subplot(gs_sH1N1[0, 0])

    # a -- trajectories

    lineprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 0.7 }
    colors = sns.husl_palette(len(x))

    pprops = { 'xticks'      : [0, 1000, 2000, 3000, 4000],
               'yticks'      : [0, 1],
               'yminorticks' : [0.25, 0.5, 0.75],
               'nudgey'      : 1.05,
               'xlabel'      : 'Time (days since Jan 1, 2009)',
               'ylabel'      : 'H3N2 HA allele\nfrequency',
               'plotprops'   : lineprops,
               'axoffset'    : 0.1,
               'theme'       : 'open' }

    xdat = [t for k in range(len(x))]
    ydat = [k for k in x]
    mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=colors, **pprops)

    # b -- selection coefficients for H3N2

    # s_ag  = dfs[0][(dfs[0]['RBS']==True) | (dfs[0]['Koel']==True)]['Selection_coefficient']
    # s_non = dfs[0][(dfs[0]['RBS']==False) & (dfs[0]['Koel']==False)]['Selection_coefficient']

    s_ag  = dfs[0][(dfs[0]['RBS']==True)]['Selection_coefficient']
    s_non = dfs[0][(dfs[0]['RBS']==False)]['Selection_coefficient']

    print('H3N2: %d RBS, %d non-RBS' % (len(s_ag), len(s_non)))
    print(st.ttest_ind(s_ag, s_non))

    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.7, edgecolor='none', orientation='vertical')

    pprops = { 'xlim'        : [-0.005, 0.015],
               'xticks'      : [  -0.005,   0.,   0.005,   0.01,   0.015],
               'xticklabels' : [r'$-0.5$', r'$0$', r'$0.5$', r'$1$', r'$1.5$'],
               'ylim'        : [0.001, 1],
               'yticks'      : [0.001, 0.01, 0.1, 1],
               'ylabel'      : 'Frequency',
               'xlabel'      : 'H3N2 HA inferred selection coefficient (%)',
               'bins'        : np.arange(-0.005, 0.015+0.001, 0.0005),
               'combine'     : True,
               'plotprops'   : histprops,
               'axoffset'    : 0.1,
               'theme'       : 'open' }
    
    ax_sH3N2.set_yscale('log')
    mp.hist(             ax=ax_sH3N2, x=[s_non], colors=[LCOLOR], **pprops)
    mp.plot(type='hist', ax=ax_sH3N2, x=[s_ag], colors=[C_MPL], **pprops)

    # c -- selection coefficients for H1N1
    s_ag  = dfs[1][(dfs[1]['RBS']==True)]['Selection_coefficient']
    s_non = dfs[1][(dfs[1]['RBS']==False)]['Selection_coefficient']

    print('H1N1: %d RBS, %d non-RBS' % (len(s_ag), len(s_non)))
    print(st.ttest_ind(s_ag, s_non))

    pprops = { 'xlim'        : [-0.005, 0.015],
               'xticks'      : [  -0.005,   0.,   0.005,   0.01,   0.015],
               'xticklabels' : [r'$-0.5$', r'$0$', r'$0.5$', r'$1$', r'$1.5$'],
               'ylim'        : [0.001, 1],
               'yticks'      : [0.001, 0.01, 0.1, 1],
               'ylabel'      : 'Frequency',
               'xlabel'      : 'H1N1 HA inferred selection coefficient (%)',
               'bins'        : np.arange(-0.005, 0.015+0.001, 0.0005),
               'combine'     : True,
               'plotprops'   : histprops,
               'axoffset'    : 0.1,
               'theme'       : 'open' }
    
    ax_sH1N1.set_yscale('log')
    mp.hist(             ax=ax_sH1N1, x=[s_non], colors=[LCOLOR], **pprops)
    mp.plot(type='hist', ax=ax_sH1N1, x=[s_ag], colors=[C_MPL], **pprops)

    # labels

    ax_traj.text(box_traj['left']-0.12, box_traj['top']+0.03, 'A', transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_sH3N2.text(box_sH3N2['left']-0.12, box_sH3N2['top']+0.03, 'B', transform=fig.transFigure, **DEF_SUBLABELPROPS)
    # ax_sH1N1.text(box_sH1N1['left']-0.05, box_sH1N1['top'], 'C', transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_prediction_grant(pfix_files, s_files, fig_name='fig-overview.pdf'):
    """ Plot H3N2 HA trajectories, selection coefficients based on whether or not mutations are in antigenic sites. """

    # Read in data
    dfs = [pd.read_csv(pfix_file, comment='#', memory_map=True) for pfix_file in pfix_files]
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = 7.5
    goldh = w
    fig   = plt.figure(figsize=(w, goldh))

    box = dict(left=0.15, right=0.95, top=0.95, bottom=0.15)
    gs  = gridspec.GridSpec(1, 1, **box)
    ax  = plt.subplot(gs[0, 0])

    lineprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 1.0 }
    dashprops = { 'lw': SIZELINE*1.0, 'linestyle': ':', 'alpha': 0.3 }
    
    pprops = { 'colors':      [C_BEN, C_BEN_LT, C_NEU, C_DEL_LT, C_DEL],
               'xlim':        [0, 1.02],
               'xticks':      [0, 0.5, 1],
               'xticklabels': ['0', '0.5', '1'],
               'xlabel':      'Mutation frequency',
               'ylim':        [0,  1.02],
               'yticks':      [0, 0.5, 1],
               'yticklabels': ['0', '0.5', '1'],
               'ylabel':      'Fixation probability',
               'axoffset':    0,
               'hide':        ['top', 'right'],
               'theme':       'open' }

    xs = [df['Frequency'] for i in range(5)]
    ys = [df['Top25_score'], df['Top_score'], df['Null'], df['Bottom_score'], df['Bottom25_score']]

    mp.plot(type='line', ax=ax, x=xs, y=ys, plotprops=lineprops, **pprops)

    pprops['colors'] = [BKCOLOR]
    mp.line(ax=ax, x=[[0, 1]], y=[[0, 1]], plotprops=dashprops, **pprops)
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = 3.35
    goldh = 1.2*w
    fig   = plt.figure(figsize=(w, goldh))

    dy = 0.12
    box_top = 0.94
    box_bot = 0.11
    box_y = (box_top - box_bot - 2*dy) / 3
    box_traj = dict(left=0.18, right=0.96, top=box_top, bottom=box_top - box_y)
    box_sH3N2 = dict(left=0.18, right=0.96, top=box_top - 1*box_y - 1*dy, bottom=box_top - 2*box_y - 1*dy)
    box_sH1N1 = dict(left=0.18, right=0.96, top=box_top - 2*box_y - 2*dy, bottom=box_top - 3*box_y - 2*dy)

    gs_traj  = gridspec.GridSpec(1, 1, **box_traj)
    gs_sH3N2 = gridspec.GridSpec(1, 1, **box_sH3N2)
    gs_sH1N1 = gridspec.GridSpec(1, 1, **box_sH1N1)

    ax_traj  = plt.subplot(gs_traj[0, 0])
    ax_sH3N2 = plt.subplot(gs_sH3N2[0, 0])
    ax_sH1N1 = plt.subplot(gs_sH1N1[0, 0])

    # a -- trajectories

    lineprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 0.7 }
    colors = sns.husl_palette(len(x))

    pprops = { 'xticks'      : [0, 1000, 2000, 3000, 4000],
               'yticks'      : [0, 1],
               'yminorticks' : [0.25, 0.5, 0.75],
               'nudgey'      : 1.05,
               'xlabel'      : 'Time (days since Jan 1, 2009)',
               'ylabel'      : 'H3N2 HA allele\nfrequency',
               'plotprops'   : lineprops,
               'axoffset'    : 0.1,
               'theme'       : 'open' }

    xdat = [t for k in range(len(x))]
    ydat = [k for k in x]
    mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=colors, **pprops)

    # b -- selection coefficients for H3N2

    # s_ag  = dfs[0][(dfs[0]['RBS']==True) | (dfs[0]['Koel']==True)]['Selection_coefficient']
    # s_non = dfs[0][(dfs[0]['RBS']==False) & (dfs[0]['Koel']==False)]['Selection_coefficient']

    s_ag  = dfs[0][(dfs[0]['RBS']==True)]['Selection_coefficient']
    s_non = dfs[0][(dfs[0]['RBS']==False)]['Selection_coefficient']

    print('H3N2: %d RBS, %d non-RBS' % (len(s_ag), len(s_non)))
    print(st.ttest_ind(s_ag, s_non))

    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.7, edgecolor='none', orientation='vertical')

    pprops = { 'xlim'        : [-0.005, 0.015],
               'xticks'      : [  -0.005,   0.,   0.005,   0.01,   0.015],
               'xticklabels' : [r'$-0.5$', r'$0$', r'$0.5$', r'$1$', r'$1.5$'],
               'ylim'        : [0.001, 1],
               'yticks'      : [0.001, 0.01, 0.1, 1],
               'ylabel'      : 'Frequency',
               'xlabel'      : 'H3N2 HA inferred selection coefficient (%)',
               'bins'        : np.arange(-0.005, 0.015+0.001, 0.0005),
               'combine'     : True,
               'plotprops'   : histprops,
               'axoffset'    : 0.1,
               'theme'       : 'open' }
    
    ax_sH3N2.set_yscale('log')
    mp.hist(             ax=ax_sH3N2, x=[s_non], colors=[LCOLOR], **pprops)
    mp.plot(type='hist', ax=ax_sH3N2, x=[s_ag], colors=[C_MPL], **pprops)

    # c -- selection coefficients for H1N1
    s_ag  = dfs[1][(dfs[1]['RBS']==True)]['Selection_coefficient']
    s_non = dfs[1][(dfs[1]['RBS']==False)]['Selection_coefficient']

    print('H1N1: %d RBS, %d non-RBS' % (len(s_ag), len(s_non)))
    print(st.ttest_ind(s_ag, s_non))

    pprops = { 'xlim'        : [-0.005, 0.015],
               'xticks'      : [  -0.005,   0.,   0.005,   0.01,   0.015],
               'xticklabels' : [r'$-0.5$', r'$0$', r'$0.5$', r'$1$', r'$1.5$'],
               'ylim'        : [0.001, 1],
               'yticks'      : [0.001, 0.01, 0.1, 1],
               'ylabel'      : 'Frequency',
               'xlabel'      : 'H1N1 HA inferred selection coefficient (%)',
               'bins'        : np.arange(-0.005, 0.015+0.001, 0.0005),
               'combine'     : True,
               'plotprops'   : histprops,
               'axoffset'    : 0.1,
               'theme'       : 'open' }
    
    ax_sH1N1.set_yscale('log')
    mp.hist(             ax=ax_sH1N1, x=[s_non], colors=[LCOLOR], **pprops)
    mp.plot(type='hist', ax=ax_sH1N1, x=[s_ag], colors=[C_MPL], **pprops)

    # labels

    ax_traj.text(box_traj['left']-0.12, box_traj['top']+0.03, 'A', transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_sH3N2.text(box_sH3N2['left']-0.12, box_sH3N2['top']+0.03, 'B', transform=fig.transFigure, **DEF_SUBLABELPROPS)
    # ax_sH1N1.text(box_sH1N1['left']-0.05, box_sH1N1['top'], 'C', transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


#___ SLIDES

def plot_surveillance_slides(surv_file, t_start=1950, t_end=2025):
    """ Plot number of sampled viruses over time. """

    # Read in data
    df = pd.read_csv(surv_file, comment='#', memory_map=True)
    x = df['date']
    y = df['count']
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SLIDE_WIDTH
    goldh = w / 2.5
    fig   = plt.figure(figsize=(w, goldh))

    box = dict(left=0.12, right=0.95, top=0.95, bottom=0.15)
    gs  = gridspec.GridSpec(1, 1, **box)
    ax  = plt.subplot(gs[0, 0])

    lineprops  = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 1.0 }
    alphaprops = { 'lw': SIZELINE*1.0, 'linestyle': '-', 'alpha': 0.3 }
    
    pprops = { 'colors':   [C_SLIDE_1],
               'xlim':     [t_start, t_end],
               'xticks':   [1950, 1965, 1980, 1995, 2010, 2025],
               'xminorticks': [1955, 1960, 1970, 1975, 1985, 1990, 2000, 2005, 2015, 2020],
               'xlabel':   'Sample year',
               'ylim':     [1e2,  1e6],
               'yticks':   [1e2, 1e3, 1e4, 1e5, 1e6],
               'ylabel':   'Cumulative number of influenza\nvirus sequences in GISAID',
               'logy':     True,
               'axoffset': 0,
               'hide':     ['top', 'right'],
               'theme':    'open' }
    
    mp.plot(type='line', ax=ax, x=[x], y=[y], plotprops=lineprops, **pprops)

    # SAVE FIGURE

    plt.savefig('%s/fig-gisaid-slides%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_overview_schematic_true_slides(freq_file, fig_name='fig-overview-true-slides.pdf'):
    """ Plot simulated trajectories """

    # Read in data
    df = pd.read_csv(freq_file, comment='#', memory_map=True)
    vars = df['variant'].unique()
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SLIDE_WIDTH/3.5
    goldh = 2
    fig   = plt.figure(figsize=(w, goldh))

    box_top  = 0.95
    box_bot  = 0.33
    box_traj = dict(left=0.12, right=0.93, top=box_top, bottom=box_bot)
    gs_traj  = gridspec.GridSpec(1, 1, **box_traj)
    ax_traj  = plt.subplot(gs_traj[0, 0])

    # a -- trajectories

    lineprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 1.0 }
    colors = sns.husl_palette(len(vars))

    pprops = { 'xticks'      : [0, 100, 200, 300],
               'yticks'      : [0, 1],
               'yminorticks' : [0.25, 0.5, 0.75],
               'ylim'        : [0, 1.05],
               'xlabel'      : 'Time',
            #    'ylabel'      : 'Mutation frequency',
               'plotprops'   : lineprops,
               'axoffset'    : 0.1,
               'theme'       : 'open' }

    xdat = [df['time'].unique() for v in vars]
    ydat = [df[df['variant']==v]['frequency'].values for v in vars]
    mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=colors, **pprops)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_overview_schematic_sim_slides(freq_file, fig_name='fig-overview-sim-slides.pdf'):
    """ Plot simulated trajectories """

    # Read in data
    df = pd.read_csv(freq_file, comment='#', memory_map=True)
    vars = df['variant'].unique()
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SLIDE_WIDTH/2.5
    goldh = 2
    fig   = plt.figure(figsize=(w, goldh))

    box_top  = 0.95
    box_bot  = 0.33
    box_traj = dict(left=0.12, right=0.93, top=box_top, bottom=box_bot)
    gs_traj  = gridspec.GridSpec(1, 1, **box_traj)
    ax_traj  = plt.subplot(gs_traj[0, 0])

    # a -- trajectories

    lineprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 0.5 }
    colors = sns.husl_palette(len(vars))

    pprops = { 'xticks'      : [0, 100, 200, 300, 400, 500],
               'yticks'      : [0, 1],
               'yminorticks' : [0.25, 0.5, 0.75],
               'ylim'        : [0, 1.05],
               'xlabel'      : 'Time',
            #    'ylabel'      : 'Mutation frequency',
               'plotprops'   : lineprops,
               'axoffset'    : 0.1,
               'theme'       : 'open' }

    # Plot each simulation separately
    sims = df['sim'].unique()
    for sim in sims[:-1]:
        df_sim = df[df['sim'] == sim]
        xdat = [df_sim['time'].unique() for v in vars]
        ydat = [df_sim[df_sim['variant']==v]['frequency'].values for v in vars]
        mp.line(ax=ax_traj, x=xdat, y=ydat, colors=colors, **pprops)

    df_sim = df[df['sim'] == sims[-1]]
    xdat = [df_sim['time'].unique() for v in vars]
    ydat = [df_sim[df_sim['variant']==v]['frequency'].values for v in vars]
    mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=colors, **pprops)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_overview_schematic_true_and_sim_slides(freq_file_true, freq_file_sim, fig_name='fig-overview-sim-prediction-slides.pdf'):
    """ Plot simulated trajectories """

    # Read in data
    df_true = pd.read_csv(freq_file_true, comment='#', memory_map=True)
    df_sim = pd.read_csv(freq_file_sim, comment='#', memory_map=True)

    vars = df_true['variant'].unique()
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SLIDE_WIDTH
    goldh = 3.0
    fig   = plt.figure(figsize=(w, goldh))

    box_top  = 0.95
    box_bot  = 0.22
    box_traj = dict(left=0.06, right=0.96, top=box_top, bottom=box_bot)
    gs_traj  = gridspec.GridSpec(1, 1, **box_traj)
    ax_traj  = plt.subplot(gs_traj[0, 0])

    # a -- trajectories

    lineprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 1.0 }
    colors = sns.husl_palette(len(vars))

    pprops = { 'xticks'      : [0, 100, 200, 300, 400, 500],
               'yticks'      : [0, 1],
               'yminorticks' : [0.25, 0.5, 0.75],
               'ylim'        : [0, 1.05],
               'xlabel'      : 'Time (generations)',
               'ylabel'      : 'Mutation frequency',
               'plotprops'   : lineprops,
               'axoffset'    : 0.1,
               'theme'       : 'open' }

    # Plot true trajectories
    xdat = [df_true['time'].unique() for v in vars]
    ydat = [df_true[df_true['variant']==v]['frequency'].values for v in vars]
    mp.line(ax=ax_traj, x=xdat, y=ydat, colors=colors, **pprops)

    lineprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 0.5 }
    pprops['plotprops'] = lineprops

    # Plot each simulation separately
    sims = df_sim['sim'].unique()
    for sim in sims[:-1]:
        df_sim_temp = df_sim[df_sim['sim'] == sim]
        xdat = [df_sim_temp['time'].unique() for v in vars]
        ydat = [df_sim_temp[df_sim_temp['variant']==v]['frequency'].values for v in vars]
        mp.line(ax=ax_traj, x=xdat, y=ydat, colors=colors, **pprops)

    df_sim_temp = df_sim[df_sim['sim'] == sims[-1]]
    xdat = [df_sim_temp['time'].unique() for v in vars]
    ydat = [df_sim_temp[df_sim_temp['variant']==v]['frequency'].values for v in vars]
    mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=colors, **pprops)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_fix_loss_example_slides(f_fix, f_loss, f_dir, t_start=1, t_end=4000, fig_name='fig-fix-loss-example-slides.pdf'):
    """ Plot simulated trajectories """

    # Read in data
    traj_fix = np.loadtxt(f_fix)
    traj_loss = np.loadtxt(f_loss)
    traj_t = np.loadtxt(f_dir + 'full_times.txt')
    t_obs = np.loadtxt(f_dir + 'observed_times.txt')[-1] - 50
    traj_fix_sims = [np.loadtxt(f_dir + '1000_sampled_trajectories/sim_traj_fix_mut_%d.txt' % (i+1)) for i in range(1000)]
    traj_loss_sims = [np.loadtxt(f_dir + '1000_sampled_trajectories/sim_traj_lost_mut_%d.txt' % (i+1)) for i in range(1000)]

    # Each simulation stops when the frequencies reach 0 or 1, so pad the arrays to the same length using the last value.
    max_len = max([len(traj_fix_sims[i]) for i in range(len(traj_fix_sims))])
    traj_fix_sims = [np.pad(traj_fix_sims[i], (0, max_len - len(traj_fix_sims[i])), 'edge') for i in range(len(traj_fix_sims))]
    traj_loss_sims = [np.pad(traj_loss_sims[i], (0, max_len - len(traj_loss_sims[i])), 'edge') for i in range(len(traj_loss_sims))]

    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SLIDE_WIDTH
    goldh = w/2.5
    fig   = plt.figure(figsize=(w, goldh))

    box_top  = 0.95
    box_bot  = 0.12
    box_traj = dict(left=0.06, right=0.96, top=box_top, bottom=box_bot)
    gs_traj  = gridspec.GridSpec(1, 1, **box_traj)
    ax_traj  = plt.subplot(gs_traj[0, 0])

    # a -- trajectories

    xticks      = np.linspace(t_start, t_end, num=10)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    xticklabels = [f'{i.year}-{i.month}' for i in xticklabels]

    lineprops = { 'lw': SIZELINE*2.5, 'linestyle': '-', 'alpha': 1.0 }
    colors = [C_MPL, C_NEU]

    pprops = { 'xlim'        : [t_start, t_end],
               'xticks'      : xticks,
               'xticklabels' : xticklabels,
               'yticks'      : [0, 1],
               'yminorticks' : [0.25, 0.5, 0.75],
               'ylim'        : [0, 1.05],
            #    'xlabel'      : 'Time (generations)',
            #    'ylabel'      : 'Mutation frequency',
               'plotprops'   : lineprops,
               'axoffset'    : 0.1,
               'theme'       : 'open' }

    # Plot true trajectories
    xdat = [traj_t for k in range(2)]
    ydat = [traj_fix, traj_loss]
    mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=colors, **pprops)

    lineprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 0.5 }
    pprops['plotprops'] = lineprops

    # Plot each simulation separately
    n_sims = 10
    xdat = [t_obs + np.arange(0, len(traj_fix_sims[k]), 1) for k in range(n_sims)]
    ydat = [traj_fix_sims[k] for k in range(n_sims)]
    mp.line(ax=ax_traj, x=xdat, y=ydat, colors=[C_MPL for k in range(n_sims)], **pprops)

    xdat = [t_obs + np.arange(0, len(traj_loss_sims[k]), 1) for k in range(n_sims)]
    ydat = [traj_loss_sims[k] for k in range(n_sims)]
    mp.line(ax=ax_traj, x=xdat, y=ydat, colors=[C_NEU for k in range(n_sims)], **pprops)

    # Plot average across all simulations
    xdat = [t_obs + np.arange(0, len(traj_fix_sims[0]), 1) for k in range(2)]
    ydat = [np.mean(traj_fix_sims, axis=0), np.mean(traj_loss_sims, axis=0)]
    colors = [C_MPL, C_NEU]
    pprops['plotprops'] = { 'lw': SIZELINE*1.5, 'linestyle': '--', 'alpha': 1.0 }
    mp.line(ax=ax_traj, x=xdat, y=ydat, colors=colors, **pprops)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_frequency_slides(freq_file, deltat=0, color=None, fig_name='fig-frequency-slides.pdf'):
    """ Plot mutation frequency trajectories. """

    # Read in data
    x = np.loadtxt(freq_file)
    above_cut = np.array([np.max(x[i])>0.01 for i in range(len(x))])
    x = x[above_cut]
    not_WT = np.array([x[i][0]<0.95 for i in range(len(x))])
    x = x[not_WT]
    t = np.arange(0, len(x[0])*10, 10)
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    # single figure
    # w     = SLIDE_WIDTH
    # goldh = 3.5
    # fig   = plt.figure(figsize=(w, goldh))
    # box_top = 0.95
    # box_bot = 0.12

    # panel
    w     = 8.5
    goldh = 1.6 #3.5
    fig   = plt.figure(figsize=(w, goldh))
    box_top = 0.95
    box_bot = 0.18

    box_traj = dict(left=0.05, right=0.96, top=box_top, bottom=box_bot)
    gs_traj  = gridspec.GridSpec(1, 1, **box_traj)
    ax_traj  = plt.subplot(gs_traj[0, 0])

    # a -- trajectories

    lineprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 0.7 }
    colors = sns.husl_palette(len(x))
    if color is not None:
        colors = [color for k in range(len(x))]

    # times t are in days since Jan 1, 2009; convert to month/year for plot using datetime
    # add x axis label every 6 months
    # single figure
    # t_start = dt.datetime(2009, 1, 1)
    # t_days  = [i*3*365/2 for i in range(8)]
    # t_dates = [t_start + dt.timedelta(days=int(i)) for i in t_days]
    # t_labels = [t_date.strftime('%b %Y') for t_date in t_dates]

    # panel
    t_start = dt.datetime(2009, 1, 1)
    t_days  = [i*2*365 for i in range(6)]
    t_dates = [t_start + dt.timedelta(days=int(i)) for i in t_days]
    t_labels = [t_date.strftime('%b %Y') for t_date in t_dates]

    pprops = { 'xticks'      : t_days,
               'xticklabels' : t_labels,
               'xlim'        : [0, 4100],
               'yticks'      : [0, 1],
               'yminorticks' : [0.25, 0.5, 0.75],
               'nudgey'      : 1.05,
            #    'xlabel'      : 'Time (days since Jan 1, 2009)',
            #    'ylabel'      : 'H3N2 HA allele\nfrequency',
               'plotprops'   : lineprops,
               'axoffset'    : 0.1,
               'theme'       : 'open' }

    xdat = [t + deltat for k in range(len(x))]
    ydat = [k for k in x]
    mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=colors, **pprops)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_selection_slides(s_file, min_counts=2, color=None, fig_name='fig-selection-slides.pdf'):
    """ Plot selection coefficients. """

    # Read in data
    df = pd.read_csv(s_file, comment='#', memory_map=True)
    df = df[df['MutationFrequency']>=min_counts]
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = 3.4
    goldh = 1.6
    fig   = plt.figure(figsize=(w, goldh))

    box_top = 0.93
    box_bot = 0.18
    box_s = dict(left=0.18, right=0.94, top=box_top, bottom=box_bot)
    gs_s  = gridspec.GridSpec(1, 1, **box_s)
    ax_s  = plt.subplot(gs_s[0, 0])

    # b -- selection coefficients for H3N2

    # s_ag  = dfs[0][(dfs[0]['RBS']==True) | (dfs[0]['Koel']==True)]['Selection_coefficient']
    # s_non = dfs[0][(dfs[0]['RBS']==False) & (dfs[0]['Koel']==False)]['Selection_coefficient']

    # s_ag  = df[(df['RBS']==True)]['Selection_coefficient']
    # s_non = df[(df['RBS']==False)]['Selection_coefficient']
    s_all = df['Selection_coefficient']

    # print('H3N2: %d RBS, %d non-RBS' % (len(s_ag), len(s_non)))
    # print(st.ttest_ind(s_ag, s_non))

    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=1, edgecolor='none', orientation='vertical')

    pprops = { 'xlim'        : [-0.005, 0.015],
               'xticks'      : [  -0.005,   0.,   0.005,   0.01,   0.015],
               'xticklabels' : [r'$-0.5$', r'$0$', r'$0.5$', r'$1$', r'$1.5$'],
               'ylim'        : [0.001, 1],
               'yticks'      : [0.001, 0.01, 0.1, 1],
               'ylabel'      : 'Frequency',
               'xlabel'      : 'Inferred selection coefficient (%)',
               'bins'        : np.arange(-0.005, 0.015+0.001, 0.0005),
               'combine'     : True,
               'plotprops'   : histprops,
               'axoffset'    : 0.1,
               'theme'       : 'open' }
    
    ax_s.set_yscale('log')
    # mp.hist(             ax=ax_s, x=[s_non], colors=[LCOLOR], **pprops)
    # mp.plot(type='hist', ax=ax_s, x=[s_ag], colors=[C_MPL], **pprops)
    c_plot = [C_MPL] if color is None else [color]
    mp.plot(type='hist', ax=ax_s, x=[s_all], colors=c_plot, **pprops)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_selection_subset_slides(s_file, min_counts=2, subset='RBS', color=None, fig_name='fig-selection-slides.pdf'):
    """ Plot selection coefficients. """

    # Read in data
    df = pd.read_csv(s_file, comment='#', memory_map=True)
    df = df[df['MutationFrequency']>=min_counts]
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = 5.25
    goldh = w/GOLDR
    fig   = plt.figure(figsize=(w, goldh))

    box_top = 0.95
    box_bot = 0.20
    box_s = dict(left=0.18, right=0.95, top=box_top, bottom=box_bot)
    gs_s  = gridspec.GridSpec(1, 1, **box_s)
    ax_s  = plt.subplot(gs_s[0, 0])

    # b -- selection coefficients

    # s_ag  = dfs[0][(dfs[0]['RBS']==True) | (dfs[0]['Koel']==True)]['Selection_coefficient']
    # s_non = dfs[0][(dfs[0]['RBS']==False) & (dfs[0]['Koel']==False)]['Selection_coefficient']

    s_sub = df[(df[subset]==True)]['Selection_coefficient']
    s_non = df[(df[subset]==False)]['Selection_coefficient']
    # s_all = df['Selection_coefficient']

    print(st.ttest_ind(s_sub, s_non))

    # print('H3N2: %d RBS, %d non-RBS' % (len(s_ag), len(s_non)))
    # print(st.ttest_ind(s_ag, s_non))

    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.7, edgecolor='none', orientation='vertical')

    pprops = { 'xlim'        : [-0.005, 0.015],
               'xticks'      : [  -0.005,   0.,   0.005,   0.01,   0.015],
               'xticklabels' : [r'$-0.5$', r'$0$', r'$0.5$', r'$1$', r'$1.5$'],
               'ylim'        : [0.001, 1],
               'yticks'      : [0.001, 0.01, 0.1, 1],
               'ylabel'      : 'Frequency',
               'xlabel'      : 'Inferred selection coefficient (%)',
               'bins'        : np.arange(-0.005, 0.015+0.001, 0.0005),
               'combine'     : True,
               'plotprops'   : histprops,
               'axoffset'    : 0.1,
               'theme'       : 'open' }
    
    ax_s.set_yscale('log')
    c_plot = [C_MPL] if color is None else [color]
    mp.hist(             ax=ax_s, x=[s_non], colors=[LCOLOR], **pprops)
    mp.plot(type='hist', ax=ax_s, x=[s_sub], colors=c_plot, **pprops)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_emd_distribution_slides(emd_file, null_col='Null_Dist', sim_col='Medoid_Dist', color=None, fig_name='fig-emd-distribution-slides.pdf', t_min=0):
    """ Plot EMD distribution. """

    # Read in data
    df = pd.read_csv(emd_file, comment='#', memory_map=True)
    if t_min>0 and 'Obs_Time' in df:
        df = df[df['Obs_Time']>t_min]
    
    # PLOT FIGURE
    
    ## set up figure grid
    w     = 5.0
    goldh = 3.0
    fig   = plt.figure(figsize=(w, goldh))

    box_top = 0.96
    box_bot = 0.21
    box = dict(left=0.20, right=0.95, top=box_top, bottom=box_bot)
    gs  = gridspec.GridSpec(1, 1, **box)
    ax  = plt.subplot(gs[0, 0])

    # b -- EMD distribution
    emd = df[null_col]- df[sim_col]

    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=1, edgecolor='none', orientation='vertical')

    pprops = { 'xlim'        : [-5, 5],
               'xticks'      : [-4, -2, 0, 2, 4],
               'xminorticks' : [-5, -3, -1, 1, 3, 5],
            #    'ylim'        : [0.01, 1],
            #    'yticks'      : [0.01, 0.1, 1],
               'ylim'        : [0, 0.25],
               'yticks'      : [0, 0.1, 0.20],
               'yminorticks' : [0.05, 0.15, 0.25],
               'ylabel'      : 'Frequency',
               'xlabel'      : 'Improvement in EMD',
               'bins'        : np.arange(-5, 5+0.25, 0.25),
               'combine'     : True,
               'plotprops'   : histprops,
               'axoffset'    : 0.1,
               'theme'       : 'open' }
    
    # ax.set_yscale('log')
    c_plot = [C_MPL] if color is None else [color]
    mp.plot(type='hist', ax=ax, x=[emd], colors=c_plot, **pprops)

    # print('EMD statistics: max, min, mean, median')
    # print(np.max(emd), np.min(emd), np.mean(emd), np.median(emd))

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_example_frequency_slides(freq_file, fig_name='fig-example-frequency-slides.pdf'):
    """ Plot mutation frequency trajectories. """

    # Read in data
    x = np.loadtxt(freq_file)
    # above_cut = np.array([np.max(x[i])>0.01 for i in range(len(x))])
    # x = x[above_cut]
    # not_WT = np.array([x[i][0]<0.95 for i in range(len(x))])
    # x = x[not_WT]
    t = np.arange(0, len(x[0]), 1)
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SLIDE_WIDTH
    goldh = 3.5
    fig   = plt.figure(figsize=(w, goldh))

    box_top = 0.95
    box_bot = 0.12
    box_traj = dict(left=0.05, right=0.96, top=box_top, bottom=box_bot)
    gs_traj  = gridspec.GridSpec(1, 1, **box_traj)
    ax_traj  = plt.subplot(gs_traj[0, 0])

    # a -- trajectories

    lineprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 0.7 }
    colors = sns.husl_palette(len(x))

    pprops = { 'xticks'      : [0, 200, 400, 600, 800, 1000],
               'yticks'      : [0, 1],
               'yminorticks' : [0.25, 0.5, 0.75],
               'nudgey'      : 1.05,
            #    'xlabel'      : 'Time (days since Jan 1, 2009)',
            #    'ylabel'      : 'H3N2 HA allele\nfrequency',
               'plotprops'   : lineprops,
               'axoffset'    : 0.1,
               'theme'       : 'open' }

    xdat = [t for k in range(len(x))]
    ydat = [k for k in x]
    mp.plot(type='line', ax=ax_traj, x=xdat, y=ydat, colors=colors, **pprops)

    # SAVE FIGURE

    plt.savefig('%s/%s' % (FIG_DIR, fig_name), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_simulation_slides(s_true=0.04, x0=0.01, N=1000, t_pred=100, t_final=500, n_sims=30, gamma=1, seed=0):
    """ Simulate evolution with for a mutation with true selection coefficient s_true, starting at
        frequency x0 in a population of N individuals, from t=0 to t=t_pred. Then, infer the selection
        coefficient and its uncertainty, draw selection coefficients from the posterior distribution,
        and simulation n_sims trajectories from t_pred to t_final. Plot the results. """

    # Wright-Fisher simulation
    rng = np.random.default_rng(seed)
    def wf(x, s, N):
        p = x * (1 + s) / (x * (1 + s) + (1 - x))
        n = rng.binomial(N, p)
        return n / N
    
    # Initial simulation
    x_vec = [x0]
    t_vec = [0]
    for t in range(t_pred-1):
        x = wf(x_vec[-1], s_true, N)
        x_vec.append(x)
        t_vec.append(t+1)

    x_vec = np.array(x_vec)
    t_vec = np.array(t_vec)

    # Infer selection coefficient and its uncertainty
    dx = x_vec[-1]-x_vec[0]
    c_int = np.sum(x_vec * (1 - x_vec))
    s_inf = dx / (c_int + gamma)
    s_inf_err = np.sqrt(1/(c_int + gamma)/(0.1*N))

    # Draw selection coefficients from posterior distribution and simulate future trajectories
    s_vec = rng.normal(s_inf, s_inf_err, n_sims)
    x_mat = np.zeros((n_sims, t_final-t_pred))
    t_mat = np.zeros((n_sims, t_final-t_pred))
    for i in range(n_sims):
        x_mat[i,0] = x_vec[-1]
        t_mat[i,0] = t_pred
        for t in range(t_final-t_pred-1):
            x_mat[i,t+1] = wf(x_mat[i,t], s_vec[i], N)
            t_mat[i,t+1] = t_pred+t+1

    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SLIDE_WIDTH
    goldh = w / 2.5
    fig   = plt.figure(figsize=(w, goldh))

    box = dict(left=0.08, right=0.95, top=0.95, bottom=0.10)
    gs  = gridspec.GridSpec(1, 1, **box)
    ax  = plt.subplot(gs[0, 0])

    lineprops  = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 1.0 }
    alphaprops = { 'lw': SIZELINE*1.0, 'linestyle': '-', 'alpha': 0.3 }
    
    pprops = { 'colors':   [C_SLIDE],
               'xlim':     [0, t_final],
               'xticks':   [],
               'xlabel':   'Time',
               'ylim':     [-0.05,  1.05],
               'yticks':   [0, 1],
               'ylabel':   'Mutation frequency',
               'axoffset': 0,
               'hide':     ['top', 'right'],
               'theme':    'open' }
    
    mp.plot(type='line', ax=ax, x=[t_vec], y=[x_vec], plotprops=lineprops, **pprops)

    pprops['colors'] = [C_SLIDE for i in range(n_sims)]
    t_cut = 2
    mp.line(ax=ax, x=t_mat[:, t_cut:], y=x_mat[:, t_cut:], plotprops=alphaprops, **pprops)

    # SAVE FIGURE

    plt.savefig('%s/fig-simulation-slides%s' % (FIG_DIR, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def compute_AUC_slides(pred_file, auc_file, t_min=0):
    """ Compute AUC from predictions of fixation/loss. """

    # Read in data
    df = pd.read_csv(pred_file, comment='#', memory_map=True)

    # Set frequency thresholds
    f_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    deltaf = 0.05

    # p value statistics
    from MLstatkit import Delong_test

    # Compute AUC
    auc_pred = []
    auc_null = []
    auc_z = []
    auc_p = []
    for f in f_vals:
        df_f = df[(df['Empirical_frequency']>=f-deltaf) & (df['Empirical_frequency']<f+deltaf)]
        if t_min>0 and 'Time' in df_f:
            df_f = df_f[df_f['Time']>t_min]
        auc_pred.append(roc_auc_score(df_f['Fixation']=='fixed', df_f['Simulations_fixed']))
        auc_null.append(roc_auc_score(df_f['Fixation']=='fixed', df_f['Empirical_frequency']))

        # Perform DeLong's test for AUC comparison
        true = [1 if x=='fixed' else 0 for x in df_f['Fixation']]
        prob_A = df_f['Simulations_fixed']
        prob_B = df_f['Empirical_frequency']
        z_score, p_value = Delong_test(true, prob_A, prob_B, return_ci=False, return_auc=False)
        auc_z.append(z_score)
        auc_p.append(p_value)

    # Save AUC
    df_auc = pd.DataFrame({'Frequency': f_vals, 'AUC_simulation': auc_pred, 'AUC_null': auc_null, 'AUC_z': auc_z, 'AUC_p': auc_p})
    df_auc.to_csv(auc_file, index=False)


def plot_AUC_slides(auc_file, plot_null=True, plot_model=True, hide_left=False, fig_name='fig-AUC-slides'):
    """ AUC bar plots predicting fixation of mutations. """

    # Read in data
    df = pd.read_csv(auc_file, comment='#', memory_map=True)
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SLIDE_WIDTH/4
    goldh = w * 2
    fig   = plt.figure(figsize=(w, goldh))

    box = dict(left=0.25, right=0.94, top=0.95, bottom=0.10)
    gs  = gridspec.GridSpec(1, 1, **box)
    ax  = plt.subplot(gs[0, 0])

    histprops = dict(lw=AXWIDTH/2, width=0.9*0.1, align='center', orientation='vertical', edgecolor=[BKCOLOR], alpha=0.5)
    
    pprops = { 'xlim':        [0, 1],
               'xticks':      [0, 0.5, 1],
               'xticklabels': ['0', '0.5', '1'],
               'ylim':        [0., 1],
               'yticks':      [0., 0.25, 0.5, 0.75, 1],
               'plotprops':   histprops,
               'theme':       'open',
               'hide':        ['top', 'right'] }

    if hide_left:
        pprops['yticks']      = []
        pprops['yticklabels'] = []
        pprops['hide'] = ['top', 'right', 'left']

    if plot_model:
        pprops['colors'] = [C_MPL]
        mp.bar(ax=ax, x=[df['Frequency']], y=[df['AUC_simulation']], **pprops)

        print('%s\t%.2f' % (fig_name, np.mean(df['AUC_simulation']-df['AUC_null'])))

    if plot_null:
        pprops['colors'] = [C_NEU]
        mp.plot(type='bar', ax=ax, x=[df['Frequency']], y=[df['AUC_null']], **pprops)

    # SAVE FIGURE

    plt.savefig('%s/%s%s' % (FIG_DIR, fig_name, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def compute_pfix_slides(pred_file, pfix_file, t_min=0):
    """ Compute probability of fixation from predictions of fixation/loss. """

    # Read in data
    df = pd.read_csv(pred_file, comment='#', memory_map=True)

    # Set frequency thresholds
    f_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    deltaf = 0.05

    # Compute probability of fixation
    pfix_avg = []
    pfix_top50 = []
    pfix_bottom50 = []
    pfix_top25 = []
    pfix_bottom25 = []
    for f in f_vals:
        df_f = df[(df['Empirical_frequency']>=f-deltaf) & (df['Empirical_frequency']<f+deltaf)]
        if t_min>0 and 'Time' in df_f:
            df_f = df_f[df_f['Time']>t_min]
        
        pfix_avg.append(np.sum(df_f['Fixation']=='fixed')/len(df_f))

        df_f = df_f.sort_values('Simulations_fixed', ascending=False)
        fixed = np.array(df_f['Fixation']=='fixed', int)
        pfix_top50.append(np.sum(fixed[:len(fixed)//2])/(len(fixed)//2))
        pfix_bottom50.append(np.sum(fixed[len(fixed)//2:])/(len(fixed)//2))
        pfix_top25.append(np.sum(fixed[:len(fixed)//4])/(len(fixed)//4))
        pfix_bottom25.append(np.sum(fixed[-len(fixed)//4:])/(len(fixed)//4))

    # Save AUC
    df_pfix = pd.DataFrame({'Frequency': f_vals, 'Top_score': pfix_top50, 'Bottom_score': pfix_bottom50, 'Top25_score': pfix_top25, 'Bottom25_score': pfix_bottom25, 'Null': pfix_avg})
    df_pfix.to_csv(pfix_file, index=False)


def plot_pfix_slides(pfix_file, fig_name='fig-pfix-slides'):
    """ Curves showing probability that a mutation will fix, along with averages for
        top 50%/bottom 50% from simulations. """

    # Read in data
    df = pd.read_csv(pfix_file, comment='#', memory_map=True)
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SLIDE_WIDTH/2
    goldh = w
    fig   = plt.figure(figsize=(w, goldh))

    box = dict(left=0.15, right=0.95, top=0.95, bottom=0.15)
    gs  = gridspec.GridSpec(1, 1, **box)
    ax  = plt.subplot(gs[0, 0])

    lineprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 1.0 }
    dashprops = { 'lw': SIZELINE*1.0, 'linestyle': ':', 'alpha': 0.3 }
    
    pprops = { 'colors':      [C_BEN, C_BEN_LT, C_NEU, C_DEL_LT, C_DEL],
               'xlim':        [0, 1.02],
               'xticks':      [0, 0.5, 1],
               'xticklabels': ['0', '0.5', '1'],
               'xlabel':      'Mutation frequency',
               'ylim':        [0,  1.02],
               'yticks':      [0, 0.5, 1],
               'yticklabels': ['0', '0.5', '1'],
               'ylabel':      'Fixation probability',
               'axoffset':    0,
               'hide':        ['top', 'right'],
               'theme':       'open' }

    xs = [df['Frequency'] for i in range(5)]
    ys = [df['Top25_score'], df['Top_score'], df['Null'], df['Bottom_score'], df['Bottom25_score']]

    mp.plot(type='line', ax=ax, x=xs, y=ys, plotprops=lineprops, **pprops)

    pprops['colors'] = [BKCOLOR]
    mp.line(ax=ax, x=[[0, 1]], y=[[0, 1]], plotprops=dashprops, **pprops)

    # SAVE FIGURE

    plt.savefig('%s/%s%s' % (FIG_DIR, fig_name, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_emd_slides(emd_file, null_col='Null_Dist', sim_col='Medoid_Dist', fig_name='fig-emd-slides', t_min=0):
    """ Curves showing earthmover's distance between true and predicted/null population distributions. """

    # Read in data
    df = pd.read_csv(emd_file, comment='#', memory_map=True)
    if t_min>0 and 'Obs_Time' in df:
        df = df[df['Obs_Time']>t_min]
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SLIDE_WIDTH
    goldh = w / 1.8
    fig   = plt.figure(figsize=(w, goldh))

    box = dict(left=0.06, right=0.95, top=0.95, bottom=0.15)
    gs  = gridspec.GridSpec(1, 1, **box)
    ax  = plt.subplot(gs[0, 0])

    lineprops = { 'lw': SIZELINE*2.5, 'linestyle': '-', 'alpha': 1.0 }
    dashprops = { 'lw': SIZELINE*1.0, 'linestyle': ':', 'alpha': 0.3 }

    xticks      = np.linspace(np.min(df['Obs_Time']), np.max(df['Obs_Time'])+1, num=10)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    xticklabels = [f'{i.year}-{i.month}' for i in xticklabels]
    
    # pprops = { 'colors':      [C_MPL, BKCOLOR],
    #            'xticks':      xticks,
    #            'xticklabels': xticklabels,
    #            'xlabel':      'Time prediction made',
    #            'ylim':        [0,  10],
    #            'yticks':      [0, 5, 10],
    #         #    'ylabel':      'Distance between predicted\nand observed populations (EMD)',
    #            'axoffset':    0,
    #            'hide':        ['top', 'right'],
    #            'theme':       'open' }
    
    # xs = [df['Obs_Time'] for i in range(2)]
    # ys = [df['Medoid_Dist'], df['Null_Dist']]

    # mp.plot(type='line', ax=ax, x=xs, y=ys, plotprops=lineprops, **pprops)

    pprops = { 'colors':      [BKCOLOR],
               'xticks':      xticks,
               'xticklabels': xticklabels,
               'xlabel':      'Time prediction made',
               'ylim':        [-3.7,  3.7],
               'yticks':      [-3, -2, -1, 0, 1, 2, 3],
            #    'ylabel':      'Distance between model\nand null errors (EMD)',
               'axoffset':    0,
               'hide':        ['top', 'right'],
               'theme':       'open' }

    xs = [df['Obs_Time'] for i in range(1)]
    ys = [df[null_col]-df[sim_col]]

    mp.plot(type='line', ax=ax, x=xs, y=ys, plotprops=lineprops, **pprops)

    pprops['colors'] = [LCOLOR]
    mp.line(ax=ax, x=[[np.min(xs), np.max(xs)]], y=[[0, 0]], plotprops=dashprops, **pprops)

    # Statistics

    print(st.ttest_rel(df[sim_col], df[null_col]))
    print('Mean improvement in EMD: %.2f' % np.mean(df[null_col]-df[sim_col]))
    print('Median improvement in EMD: %.2f' % np.median(df[null_col]-df[sim_col]))

    # SAVE FIGURE

    plt.savefig('%s/%s%s' % (FIG_DIR, fig_name, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


def plot_emd_joint_slides(emd_files, null_col='Null_Dist', sim_col='Medoid_Dist', fig_name='fig-emd-joint-slides', t_min=0, stage=4, alt=False):
    """ Curves showing earthmover's distance between true and predicted/null population distributions. """

    # Read in data
    dfs = [pd.read_csv(f, comment='#', memory_map=True) for f in emd_files]
    if t_min>0 and 'Obs_Time' in dfs[0]:
        dfs = [df[df['Obs_Time']>t_min] for df in dfs]

    # Shift times
    t_ref = dt.date.fromisoformat('2009-01-01')
    for df in dfs:
        df['Time'] = [(dt.date.fromisoformat(df.iloc[i]['Obs_Date']) - t_ref).days for i in range(len(df))]
    
    # PLOT FIGURE
    
    ## set up figure grid
    
    w     = SLIDE_WIDTH
    goldh = w / 2
    fig   = plt.figure(figsize=(w, goldh))

    box = dict(left=0.06, right=0.95, top=0.95, bottom=0.15)
    gs  = gridspec.GridSpec(1, 1, **box)
    ax  = plt.subplot(gs[0, 0])

    mainprops = { 'lw': SIZELINE*2.5, 'linestyle': '-', 'alpha': 1.0 }
    backprops = { 'lw': SIZELINE*1.5, 'linestyle': '-', 'alpha': 0.4 }
    dashprops = { 'lw': SIZELINE*1.0, 'linestyle': ':', 'alpha': 0.3 }

    xticks      = np.linspace(np.min(dfs[0]['Time']), np.max(dfs[0]['Time'])+1, num=10)
    xticklabels = [(dt.timedelta(days=int(i)) + dt.datetime(2009,1,1)) for i in xticks]
    xticklabels = [f'{i.year}-{i.month}' for i in xticklabels]

    palette = sns.color_palette('PiYG', 8)
    colors = [palette[0], palette[1], palette[-1], palette[-2]]

    pprops = { 'colors':      colors,
               'xlim':        [1100, 3700],
               'xticks':      xticks,
               'xticklabels': xticklabels,
               'xlabel':      'Time prediction made',
               'ylim':        [-5,  5],
               'yticks':      [-4, -2, 0, 2, 4],
               'yminorticks': [-5, -3, -1, 1, 3, 5],
            #    'ylabel':      'Distance between model\nand null errors (EMD)',
               'axoffset':    0,
               'hide':        ['top', 'right'],
               'theme':       'open' }

    xs = [dfs[i]['Time'] for i in range(stage)]
    ys = [dfs[i][null_col]-dfs[i][sim_col] for i in range(stage)]

    special_idx = 0 if alt else stage-1

    if stage==0:
        pprops['colors'] = ['k']
        mp.plot(type='line', ax=ax, x=[[np.min(dfs[0]['Time']), np.max(dfs[0]['Time'])]], y=[[0, 0]], plotprops=dashprops, **pprops)

    else:
        for i in range(stage):
            if i==special_idx or alt:
                lineprops = mainprops
                pprops['colors'] = [colors[i]]
            else:
                lineprops = backprops
                pprops['colors'] = [LCOLOR]
            xs[i] = [xs[i]]
            ys[i] = [ys[i]]
            if i==special_idx:
                mp.plot(type='line', ax=ax, x=xs[i], y=ys[i], plotprops=lineprops, **pprops)
            else:
                mp.line(ax=ax, x=xs[i], y=ys[i], plotprops=lineprops, **pprops)

        # mp.plot(type='line', ax=ax, x=xs, y=ys, plotprops=lineprops, **pprops)
        pprops['colors'] = ['k']
        mp.line(ax=ax, x=[[np.min(xs[0]), np.max(xs[0])]], y=[[0, 0]], plotprops=dashprops, **pprops)

    if stage==4:
        print('\nMean improvement in EMD')
        labels = ['H3N2-HA', 'H3N2-NA', 'H1N1-HA', 'H1N1-NA']
        for i in range(len(dfs)):
            print('%s\t%.2f' % (labels[i], np.mean(ys[i])))
        for i in range(len(xs)):
            print(np.min(ys[i]), np.max(ys[i]))

    # SAVE FIGURE

    plt.savefig('%s/%s%s' % (FIG_DIR, fig_name, EXT), dpi = 1000, facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    plt.close(fig)


#^__ SLIDES


#___ SUPPLEMENTARY PLOTS
