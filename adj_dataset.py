import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from pathlib import Path


class AdjDataset(Dataset):

    def __init__(self, data_dir: Path | str, adj_dir: Path | str):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        if isinstance(adj_dir, str):
            adj_dir = Path(adj_dir)
        self.data_dir = data_dir
        self.adj_dir = adj_dir
        self.sample_file_names = os.listdir(self.data_dir)
        self.sample_file_names.remove('labels.pt')
        self.labels = torch.load(self.data_dir / 'labels.pt', weights_only=True)
        self.adj_file_names = os.listdir(self.adj_dir)
        self.adj_file_names.remove('labels.pt')

    def __len__(self):
        return len(self.sample_file_names)

    def __getitem__(self, idx):
        file_name = self.sample_file_names[idx]
        X = torch.load(self.data_dir / file_name, weights_only=True)
        y = self.labels[idx]
        weights_file_name = self.adj_file_names[idx]
        adj_weights = torch.load(self.adj_dir / weights_file_name, weights_only=True)

        return X, y, adj_weights



def adj_collate_fn(batch):
    inputs = torch.stack([sample[0] for sample in batch], dim=0)
    targets = [sample[1] for sample in batch]
    targets = torch.tensor(targets)
    adj_weights = torch.stack([sample[2] for sample in batch], dim=0)
    return inputs, targets, adj_weights

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