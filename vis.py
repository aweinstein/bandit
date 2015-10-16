import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


if __name__ == '__main__':
    d_map = {3:1, 8:2, 14:3, 23:4}
    df = pd.read_pickle('all_data.pkl')
    df['choices'] = df['choices'].apply(lambda x: d_map[x])
    df.reset_index(inplace=True)

    plt.close('all')
    g = sns.FacetGrid(df, col='subject', hue='cues',
                      col_wrap=6, size=1.5, ylim=(0, 5), aspect=1.5)
    g.map(plt.plot, 'choices')
    g.set(xticks=[], yticks=[1,2,3,4], yticklabels=['3', '8', '14', '23'])
    g.set_ylabels('choice')
    g.fig.tight_layout()
    plt.savefig('actions.pdf')
    plt.show()
