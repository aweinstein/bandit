import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

Fig_Dir = 'figs'
DF_Dir = 'df'

def plot_actions(cue=1):
    d_map = {3:1, 8:2, 14:3, 23:4}
    df = pd.read_pickle(os.path.join(DF_Dir, 'all_data.pkl'))
    df['choices'] = df['choices'].apply(lambda x: d_map[x])
    if cue in [1, 2, 3]:
        df = df.loc[df['cues'] == cue]
    df.reset_index(inplace=True)


    plt.close('all')
    if cue in [1, 2, 3]:
        g = sns.FacetGrid(df, col='subject',
                          col_wrap=6, size=1.5, ylim=(0, 5), aspect=1.5)
        title = 'Cue {:d}'.format(cue)
    else:
        g = sns.FacetGrid(df, col='subject', hue='cues',
                          col_wrap=6, size=1.5, ylim=(0, 5), aspect=1.5)
        title = 'All cues'

    g.map(plt.plot, 'choices')
    g.set(xticks=[], yticks=[1,2,3,4], yticklabels=['3', '8', '14', '23'])
    g.set_ylabels('choice')
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.93)
    g.fig.suptitle(title)

    if cue in [1, 2, 3]:
        subjects = df['subject'].unique()
        for ax, subject in zip(g.axes, subjects):
            df_subject = df.loc[df['subject'] == subject]
            df_subject.reset_index(inplace=True)
            df_wins = df_subject.loc[df_subject['rewards'] > 0]
            df_lose = df_subject.loc[df_subject['rewards'] < 0]
            pos_win = df_wins.loc[df_wins['subject'] == subject].index
            pos_lose = df_lose.loc[df_lose['subject'] == subject].index
            ax.eventplot(pos_win, lineoffsets=4.5, linelength=0.75,
                         linewidths=0.4)
            ax.eventplot(pos_lose, lineoffsets=4.5, linelength=0.75,
                         color='r', linewidths=0.4)
    
    if cue in [1, 2, 3]:
        fn = 'actions_{:d}.pdf'.format(cue)
    else:
        fn = 'actions_all.pdf'
    plt.savefig(os.path.join(Fig_Dir, fn))
    plt.show()
    globals().update(locals())

def plot_optimum():
    df = pd.read_pickle(os.path.join(DF_Dir, 'df_n_optimum.pkl'))
    plt.close('all')
    sns.factorplot(data=df, x='block', y='n_optimum', hue='learner', aspect=1.5)
    plt.savefig(os.path.join(Fig_Dir, 'n_optimum.pdf'))
    plt.show()

if __name__ == '__main__':
    plot_actions()
