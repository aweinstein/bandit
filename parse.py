"""Extract behavioral data from the original files.
The data for each subjet is save as a Pandas DataFrame."""

import os
import pandas as pd
from scipy.io import loadmat

Data_Dir = 'data_behavior'
DF_Dir = 'df'

def matlab_to_dataframe():
    """Extract behavioral data from mat files and create a dataframe.

    One dataframe per subject is created. The dataframe is pickled and saved.
    """
    dir_name = 'data'
    for fname in os.listdir(dir_name):
        if fname.endswith('.mat'):
            s = fname[1:3]
            print('Reading', fname, '...')
            data_ml = loadmat(os.path.join(dir_name, fname))
            data_df = {}
            columns = ('cues', 'choices', 'rewards', 'accepts')
            for column in columns:
                data_df[column] = data_ml[column].flatten()
            df = pd.DataFrame(data_df, columns=columns)
            new_names = {'cues':'cue', 'choices':'action', 'rewards':'reward'}
            df.rename(columns=new_names, inplace=True)
            df.index.name = 'trial'
            d_map = {3:0, 8:1, 14:2, 23:3}
            df['action'] = df['action'].apply(lambda x: d_map[x])
            df['cue'] = df['cue'].apply(lambda x: x - 1)
            df_file = os.path.join(Data_Dir, s + '.pkl')
            print('DataFrame saved in', df_file)
            df.to_pickle(df_file)

def concat_all_dataframes():
    print('Concatenating the subject DFs')
    pkls = os.listdir(Data_Dir)
    pkls.sort()
    subjects = [int(s[:2]) for s in pkls]
    dfs = (pd.read_pickle(os.path.join(Data_Dir,fn)) for fn in pkls)
    df_all = pd.concat(dfs, keys=subjects)
    df_all.index.set_names('subject', 0, inplace=True)
    fn = os.path.join(DF_Dir,'all_data.pkl')
    df_all.to_pickle(fn)
    print('DF save as', fn)

def get_hipomania_scores():
    fn = os.path.join(DF_Dir, 'HypomaniaData.xlsx')
    df = pd.read_excel(fn)
    hps_df = df[['PN', 'HPS']]
    hps_df = hps_df.rename(columns={'PN':'subject'})
    hps_df.to_pickle(os.path.join(DF_Dir, 'df_hps.pkl'))
