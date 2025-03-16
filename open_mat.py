



import mne
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import re
import pandas as pd

data_dir = Path('./epochs_dataframes')

def getitem(data_dir, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    file_name = os.listdir(data_dir)[idx]
    df = pd.read_csv(data_dir / file_name, index_col=0)
    #print(df)
    X = torch.tensor(df.values.mean(axis=1).shape)
    y = pd.read_csv(data_dir / 'labels.csv').iloc[idx]

    return X, y

if __name__ == '__main__':
    #print(getitem(data_dir, 3))
    print(pd.read_csv(data_dir / 'labels.csv').iloc[0])