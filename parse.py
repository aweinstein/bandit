"""Extract behavioral data from the original files.
The data for each subjet is save as a Pandas DataFrame."""

import os
import pandas as pd
from scipy.io import loadmat
import numpy as np

Data_Dir = 'data_behavior'
DF_Dir = 'df'

def matlab_to_dataframe():
    """Extract behavioral data from mat files and create a dataframe.

    One dataframe per subject is created. The dataframe is pickled and saved.
    """
    dir_name = 'data'
    data_dir = 'data_behavior'
    for fname in sorted(os.listdir(dir_name)):
        if fname.endswith('.mat') and len(fname) == 7:
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
            df_file = os.path.join(data_dir, s + '.pkl')
            print('DataFrame saved in', df_file)
            df.to_pickle(df_file)

def liamcsv_to_dataframe():
    dir_name = 'data_liam'
    for fname in os.listdir(dir_name):
        if fname.endswith('.csv'):
            if fname[0] == '1':
                subject = fname[1:3]
            else:
                subject = fname[:2]
            df = pd.read_csv(os.path.join(dir_name, fname))
            new_names = {'Cue':'cue', 'Choice':'action', 'Outcome':'reward'}
            df.rename(columns=new_names, inplace=True)
            df.index.name = 'trial'
            d_map = {3:0, 8:1, 14:2, 23:3}
            df['action'] = df['action'].apply(lambda x: d_map[x])
            df['cue'] = df['cue'].apply(lambda x: x - 1)
            data_dir = Data_Dir
            df_file = os.path.join(data_dir, str(subject) + '.pkl')
            df.to_pickle(df_file)
            print('Processing file {}, subject {}. '
                  'File saved as {}'.format(fname, subject, df_file))


def concat_all_dataframes(data_dir=None, all_data_fn='all_data.pkl'):
    if data_dir is None:
        data_dir = Data_Dir
    print('Concatenating the subject DFs')
    pkls = os.listdir(data_dir)
    pkls.sort()
    subjects = [int(s[:2]) for s in pkls]
    dfs = (pd.read_pickle(os.path.join(data_dir,fn)) for fn in pkls)
    df_all = pd.concat(dfs, keys=subjects)
    df_all.index.set_names('subject', 0, inplace=True)
    fn = os.path.join(DF_Dir, all_data_fn)
    df_all.to_pickle(fn)
    print('DF save as', fn)

def get_hipomania_scores():
    fn = os.path.join(DF_Dir, 'HypomaniaData.xlsx')
    df = pd.read_excel(fn)
    hps_df = df[['PN', 'HPS']]
    hps_df = hps_df.rename(columns={'PN':'subject'})
    thresholds = [0,18,29] # thresholds for setting low, medium, and high HPS
    levels = hps_df['HPS'].apply(lambda x: np.digitize(x, thresholds))
    hps_df['HPS_level'] = levels
    fn = os.path.join(DF_Dir, 'hps_df.pkl')
    hps_df.to_pickle(fn)
    print('HPS scores saved in', fn)

if __name__ == '__main__':
    liamcsv_to_dataframe()
    concat_all_dataframes()
