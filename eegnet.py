import torch
from torch import nn
from torch.utils.data import DataLoader
from torch_dataset import EpochsDataset


class EEGNet(nn.Module):

    def __init__(self, sfreq: float, n_channels: int, n_filters1: int,
                 depth_conv_kernel_size: int, n_spatial_filters: int,
                 pool1_kernel_size: int, dropout_chance: float):
        super().__init__()
        kernel_size1 = sfreq // 2
        n_filters2 = n_filters1 * n_spatial_filters
        self.conv1 = nn.Conv1d(n_channels, n_filters1,
                               kernel_size=kernel_size1, padding='same')
        self.batch_norm = nn.BatchNorm1d(n_filters1)
        self.depth_conv = nn.Conv1d(n_filters1, n_filters2,
                                    kernel_size=depth_conv_kernel_size,
                                    groups=n_spatial_filters, padding='valid')
        self.batch_norm2 = nn.BatchNorm1d(n_filters2)
        self.activation = nn.ELU()
        self.pooling1 = nn.AvgPool1d(pool1_kernel_size)
        self.dropout = nn.Dropout(dropout_chance)



    def forward(self, input):
        output = self.conv1(input)
        output = self.batch_norm(output)
        output = self.depth_conv(output)
        output = self.batch_norm2(output)
        output = self.activation(output)
        output = self.pooling1(output)
        output = self.dropout(output)
        return output

def train(model, train_loader):
    model.train()
    for data, target in train_loader:
        output = model(data)
        return output

def main():
    model = EEGNet(sfreq=160,
                   n_channels=64,
                   n_filters1=8,
                   depth_conv_kernel_size=1,
                   n_spatial_filters=2,
                   pool1_kernel_size=4,
                   dropout_chance=0.25)
    train_loader = DataLoader(EpochsDataset('./epochs_tensors'))
    output = train(model, train_loader)
    print(output)
    print(output.shape)

if __name__ == '__main__':
    main()



