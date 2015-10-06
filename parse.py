"""Extract behavioral data from the original files.
The data for each subjet is save as a Pandas DataFrame."""

import os
import pandas as pd
from scipy.io import loadmat


if __name__ == '__main__':
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
            df_file = os.path.join('data_behavior', s + '.pkl')
            print('DataFrame saved in', df_file)
            df.to_pickle(df_file)
