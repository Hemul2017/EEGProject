


import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from pathlib import Path
import pandas as pd
import warnings


class EpochsDataset(Dataset):

    def __init__(self, data_dir: Path | str):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        self.data_dir = data_dir
        self.sample_file_names = os.listdir(self.data_dir)
        self.sample_file_names.remove('labels.pt')
        self.labels = torch.load(self.data_dir / 'labels.pt', weights_only=True)

    def __len__(self):
        return len(self.sample_file_names)

    def __getitem__(self, idx):
        file_name = self.sample_file_names[idx]
        X = torch.load(self.data_dir / file_name, weights_only=True)
        y = self.labels[idx]

        return X, y



def collate_fn(batch):
    inputs = torch.stack([sample[0] for sample in batch], dim=0)
    targets = [sample[1] for sample in batch]
    targets = torch.tensor(targets)
    return inputs, targets

if __name__ == '__main__':
    imaginary_dataset = EpochsDataset('./epochs_tensors/imaginary')
    X, y = imaginary_dataset[17]
    print(len(imaginary_dataset))
    print(X.shape)
    train_data, test_data = random_split(imaginary_dataset, [0.8, 0.2])
    print(len(train_data.indices))
    print(len(test_data.indices))
    print(len(imaginary_dataset))
    print(y)


