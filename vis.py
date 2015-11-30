import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize

Fig_Dir = 'figs'
DF_Dir = 'df'

def plot_actions(cue=0, fn='all_data_liam.pkl'):
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
    
    if cue in [0, 1, 2]:
        fn = 'actions_{:d}_{}.pdf'.format(cue, fn)
    else:
        fn = 'actions_all_{}.pdf'.format(fn)
    plt.savefig(os.path.join(Fig_Dir, fn))
    plt.show()
    globals().update(locals())

def plot_optimum():
    df = pd.read_pickle(os.path.join(DF_Dir, 'df_n_optimum.pkl'))
    plt.close('all')
    sns.factorplot(data=df, x='block', y='n_optimum', hue='learner', aspect=2.5)
    plt.savefig(os.path.join(Fig_Dir, 'n_optimum.pdf'))
    plt.show()

def scatter_alpha_beta_hps():   
    fn = 'fit_0101_unbounded_sample_average.csv'
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
    fig = plt.scatter(x, y, c=hps, cmap=cm.jet)
    # plt.xlabel('$\alpha$')
    # plt.ylabel('$\beta$')
    plt.show()

if __name__ == '__main__':
    plot_actions()
