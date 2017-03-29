from itertools import cycle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import cm
from cycler import cycler

Fig_Dir = 'figs'
DF_Dir = 'df'

def plot_actions(cue=0, fn='all_data.pkl'):
    d_map = {3:1, 8:2, 14:3, 23:4}
    df = pd.read_pickle(os.path.join(DF_Dir, fn))
    #df['cue'] = df['cue'].apply(lambda x: d_map[x])
    if cue in [0, 1, 2]:
        df = df.loc[df['cue'] == cue]
    df.reset_index(inplace=True)

    plt.close('all')
    if cue in [0, 1, 2]:
        g = sns.FacetGrid(df, col='subject',
                          col_wrap=6, size=1.5, ylim=(0, 5), aspect=1.5)
        title = 'Cue {:d}'.format(cue)
    else:
        g = sns.FacetGrid(df, col='subject', hue='cue',
                          col_wrap=6, size=1.5, ylim=(0, 5), aspect=1.5)
        title = 'All cues'

    g.map(plt.plot, 'action')
    g.set(xticks=[], yticks=[0,1,2,3], yticklabels=['3', '8', '14', '23'])
    g.set(ylim=(-0.5, 4))
    g.set_ylabels('choice')
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.93)
    g.fig.suptitle(title)

    if cue in [0, 1, 2, 3]:
        subjects = df['subject'].unique()
        for ax, subject in zip(g.axes, subjects):
            df_subject = df.loc[df['subject'] == subject]
            df_subject.reset_index(inplace=True)
            df_wins = df_subject.loc[df_subject['reward'] > 0]
            df_lose = df_subject.loc[df_subject['reward'] < 0]
            pos_win = df_wins.loc[df_wins['subject'] == subject].index
            pos_lose = df_lose.loc[df_lose['subject'] == subject].index
            ax.eventplot(pos_win, lineoffsets=3.5, linelength=0.75,
                         linewidths=0.4)
            ax.eventplot(pos_lose, lineoffsets=3.5, linelength=0.75,
                         color='r', linewidths=0.4)
    fn = os.path.splitext(fn)[0]
    if cue in [0, 1, 2]:
        fn = 'actions_{:d}_{}.pdf'.format(cue, fn)
    else:
        fn = 'actions_all_{}.pdf'.format(fn)
    fn = os.path.join(Fig_Dir, fn)
    plt.savefig(fn)
    print('Figure saved as', fn)
    plt.show()
    globals().update(locals())

def plot_optimum():
    df = pd.read_pickle(os.path.join(DF_Dir, 'df_n_optimum.pkl'))
    plt.close('all')
    sns.factorplot(data=df, x='block', y='n_optimum', hue='learner', aspect=2.5)
    plt.savefig(os.path.join(Fig_Dir, 'n_optimum.pdf'))
    plt.show()

def scatter_alpha_beta_hps():
    fn = 'fit_constant_step_size_0101_bounded.csv'
    df_ab = pd.read_csv(os.path.join(DF_Dir, fn))
    df_hps = pd.read_pickle(os.path.join(DF_Dir, 'df_hps.pkl'))
    df_hps['HPS_q'] = df_hps['HPS'].apply(lambda x: np.digitize(x, [0,18,29]))
    df_ab = df_ab.merge(df_hps, on='subject', how='left')
    # x_key, y_key = '0_alpha', '1_alpha'
    x_key, y_key = '0_beta', '1_beta'
    x_status, y_status = '0_status', '1_status'
    df_ab = df_ab[(df_ab[x_status]==0) & (df_ab[y_status]==0)]
    x, y = df_ab[x_key], df_ab[y_key]
    hps = df_ab['HPS_q']
    plt.close('all')
    plt.scatter(x, y, c=hps, cmap=cm.jet)
    # plt.xlabel('$\alpha$')
    # plt.ylabel('$\beta$')
    plt.show()

def plot_simple_bandit(df):
    """Plot the trials of a two state bandit.

    The df must have columns 'action', 'reward', ('Q(0)', and 'Q(1)') or (pi(0)
    and pi(1).
    """
    _, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
    pos_zero = df.loc[df['action'] == 0].index
    pos_one = df.loc[df['action'] == 1].index
    ax0.eventplot(pos_zero, linewidths=1.5, lineoffsets=2.5, colors=['C1'],
                  label='L')
    ax0.eventplot(pos_one, linewidths=1.5, lineoffsets=2.5, colors=['C2'],
                  label='R')

    colors=['C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for r, c in zip(np.sort(df['reward'].unique()), cycle(colors)):
        pos = df.loc[df['reward'] == r].index
        ax0.eventplot(pos, linewidths=1.5, label=str(r), color=[c])

    ax0.set_yticks([1, 2.5])
    ax0.set_yticklabels(['reward', 'action'])
    ax0.legend(loc='upper right', frameon=True)

    # Plot values or policies
    if 'Q(0)' in df.columns:
        ax1.plot(df['Q(0)'], label='Q(0)')
        ax1.plot(df['Q(1)'], label='Q(1)')
    elif 'pi(0)' in df.columns:
        ax1.plot(df['pi(0)'], label='pi(0)')
        ax1.plot(df['pi(1)'], label='pi(1)')

    ax1.legend(loc='upper right', frameon=True)
    ax1.set_xlabel('trial')
    plt.show()


if __name__ == '__main__':
    plot_actions()
