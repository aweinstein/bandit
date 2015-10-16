"""Extract behavioral data from the original files.
The data for each subjet is save as a Pandas DataFrame."""

import os
import pandas as pd
from scipy.io import loadmat

DF_Dir = 'data_behavior'

def matlab_to_dataframe():
    """Extract behavioral data from mat files and create a dataframe.

    One dataframe per subject is created. The dataframe is pickled and saved.
    """
    dir_name = 'data'
    for fname in os.listdir(dir_name):
        if fname.endswith('.mat'):
            s = fname[1:3]
            print('Reading', fname, '...')
            data = loadmat(os.path.join(dir_name, fname))
            columns = ('cues', 'choices', 'rewards', 'accepts')
            for column in columns:
                data[column] = data[column].flatten()
            df = pd.DataFrame(data, columns=columns)
            df.index.name = 'trial'
            df_file = os.path.join(DF_Dir, s + '.pkl')
            print('DataFrame saved in', df_file)
            df.to_pickle(df_file)

def concat_all_dataframes():
    pkls = os.listdir(DF_Dir)
    pkls.sort()
    subjects = [int(s[:2]) for s in pkls]
    dfs = (pd.read_pickle(os.path.join(DF_Dir,fn)) for fn in pkls)
    df_all = pd.concat(dfs, keys=subjects)
    df_all.index.set_names('subject', 0, inplace=True)
    df_all.to_pickle('all_data.pkl')

