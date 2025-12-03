


import argparse
from pathlib import Path
import csv

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch_dataset import EpochsDataset, collate_fn
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from adj_dataset import AdjDataset, adj_collate_fn


class GNN(nn.Module):
    def __init__(self, device):
        super(GNN, self).__init__()
        self.cnn = nn.Conv1d(in_channels=64, out_channels=4*64, kernel_size=5, stride=1, groups=4)
        self.batch_norm = nn.BatchNorm1d(num_features=22)
        self.pool = nn.MaxPool1d(kernel_size=5, stride=5)
        self.graph_conv = gnn.SAGEConv(in_channels=653, out_channels=653, aggr='max')
        self.activation = nn.GELU()
        self.fc1 = nn.Linear(653, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 4)
        self.adj_matrix = read_adj_matrix('./info/bci_adj_matrix.pt')
        self.adj_matrix = self.adj_matrix.to(device)
        self.batch = torch.tensor([0] * 116).to(device)

    def forward(self, input):
        #mean = torch.mean(input, dim=(0, 2))[torch.newaxis, :, torch.newaxis]
        #std = torch.std(input, dim=(0, 2))[torch.newaxis, :, torch.newaxis]
        #input = (input - mean) / std
        output = self.cnn(input)
        #output = self.pool(output)
        #output = self.batch_norm(output)
        output = self.graph_conv(x=output, edge_index=self.adj_matrix)
        #output = self.batch_norm(output)
        output = self.graph_conv(x=output, edge_index=self.adj_matrix)
        #output = self.batch_norm(output)
        output = self.graph_conv(x=output, edge_index=self.adj_matrix)
        #output = self.batch_norm(output)
        output = gnn.pool.global_mean_pool(output, self.batch)
        output = output.squeeze()
        output = self.fc1(output)
        output = self.activation(output)
        output = self.fc2(output)
        output = self.activation(output)
        output = self.fc3(output)
        output = F.softmax(output, dim=1)
        return output

class GNN2(nn.Module):
    def __init__(self, device):
        super(GNN2, self).__init__()
        N = 657
        self.cnn1 = gnn.ChebConv(in_channels=N, out_channels=N, K=3)
        self.cnn2 = gnn.ChebConv(in_channels=N, out_channels=N, K=3)
        self.cnn3 = gnn.ChebConv(in_channels=N, out_channels=N, K=3)
        self.cnn4 = gnn.ChebConv(in_channels=N, out_channels=N, K=3)
        self.cnn5 = gnn.ChebConv(in_channels=N, out_channels=N, K=3)
        self.cnn6 = gnn.ChebConv(in_channels=N, out_channels=N, K=3)
        self.batch_norm1 = nn.LazyBatchNorm1d()
        self.batch_norm2 = nn.LazyBatchNorm1d()
        self.batch_norm3 = nn.LazyBatchNorm1d()
        self.batch_norm4 = nn.LazyBatchNorm1d()
        self.batch_norm5 = nn.LazyBatchNorm1d()
        self.batch_norm6 = nn.LazyBatchNorm1d()
        self.batch_norm_fc1 = nn.LazyBatchNorm1d()
        self.batch_norm_fc2 = nn.LazyBatchNorm1d()
        self.batch_norm_fc3 = nn.LazyBatchNorm1d()
        self.activation = nn.Softplus()
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 4)
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self.flatten = nn.Flatten()
        self.adj_matrix = read_adj_matrix('./info/physionet_adj_matrix.pt')
        self.adj_matrix = self.adj_matrix.to(device)
        self.adj_matrix_list = []
        self.clusters = []

    def forward(self, input, adj_weights):

        #mean = torch.mean(input, dim=(0, 2))[torch.newaxis, :, torch.newaxis]
        #std = torch.std(input, dim=(0, 2))[torch.newaxis, :, torch.newaxis]
        #input = (input - mean) / std

        batch_size = len(input)

        adj_matrix = self.adj_matrix

        adj_matrix = torch.hstack(([adj_matrix] * batch_size))

        input = input.reshape(-1, 657)

        batch = [i for i in range(batch_size) for _ in range(64)]

        adj_weights = self.get_adj_weights(adj_matrix, adj_weights, batch_size)

        output = self.cnn1(input, edge_index=adj_matrix, edge_weight=adj_weights, batch=batch)
        output = self.activation(output)
        output = self.batch_norm1(output)
        #output, adj_matrix = self.pool(output, 0, adj_matrix, batch_size)

        """

        output = self.cnn2(output, edge_index=adj_matrix, batch=batch)
        output = self.activation(output)
        output = self.batch_norm2(output)
        #output, adj_matrix = self.pool(output, 1, adj_matrix, batch_size)

        output = self.cnn3(output, edge_index=adj_matrix, batch=batch)
        output = self.activation(output)
        output = self.batch_norm3(output)
        #output, adj_matrix = self.pool(output, 2, adj_matrix, batch_size)

        output = self.cnn4(output, edge_index=adj_matrix, batch=batch)
        output = self.activation(output)
        output = self.batch_norm4(output)
        #output, adj_matrix = self.pool(output, 3, adj_matrix, batch_size)

        output = self.cnn5(output, edge_index=adj_matrix, batch=batch)
        output = self.activation(output)
        output = self.batch_norm5(output)
        #output, adj_matrix = self.pool(output, 4, adj_matrix, batch_size)

        output = self.cnn6(output, edge_index=adj_matrix, batch=batch)
        output = self.activation(output)
        output = self.batch_norm6(output)
        #output, adj_matrix = self.pool(output, 5, adj_matrix, batch_size)
        
        """

        #output = output.reshape(batch_size, len(torch.unique(self.clusters[-1])), 657)


        output = output.reshape(batch_size, 64, 657)

        output = self.flatten(output)
        output = self.fc1(output)
        output = self.activation(output)
        output = self.dropout1(output)
        output = self.batch_norm_fc1(output)

        output = self.fc2(output)
        output = self.activation(output)
        output = self.dropout2(output)
        output = self.batch_norm_fc2(output)

        output = self.fc3(output)
        output = self.batch_norm_fc3(output)
        output = self.activation(output)

        output = F.softmax(output, dim=1)
        return output


    def pool(self, output, layer_idx, adj_matrix, batch_size):
        pooled_output = []
        if len(self.clusters) <= layer_idx:
            new_clusters = gnn.pool.graclus(edge_index=adj_matrix)
            self.clusters.append(new_clusters)
        clusters = self.clusters[layer_idx]

        if layer_idx > 0:
            n_clusters = len(torch.unique(self.clusters[layer_idx-1]))
        else:
            n_clusters = 64
        output = output.reshape(batch_size, n_clusters, 657)

        for batch_idx in range(len(output)):
            data = Data(x=output[batch_idx], edge_index=adj_matrix)
            data = gnn.pool.max_pool(clusters, data)
            pooled_output.append(data.x.tolist())

        if len(self.adj_matrix_list) <= layer_idx:
            self.adj_matrix_list.append(data.edge_index)

        output = torch.tensor(pooled_output)
        output = output.reshape(-1, 657)
        adj_matrix = self.adj_matrix_list[layer_idx]
        adj_matrix = torch.hstack(([adj_matrix] * batch_size))

        return output, adj_matrix

    def get_adj_weights(self, adj_matrix, adj_weights, batch_size):
        n_edges_per_batch = adj_matrix.shape[1] // batch_size
        adj_matrix = adj_matrix[:, :n_edges_per_batch]
        adj_weights = adj_weights.reshape(-1, 64, 64)
        adj_weights = torch.maximum(adj_weights, adj_weights.swapaxes(1, 2))

        adj_weights_list = []
        for i, k in adj_matrix.T:
            adj_weights_list.append(adj_weights[:, i, k])

        adj_weights = torch.vstack(adj_weights_list).T
        adj_weights = adj_weights.reshape(-1)

        return adj_weights



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target, adj_weights) in enumerate(train_loader):
        data, target, adj_weights = data.to(device), target.to(device), adj_weights.to(device)
        optimizer.zero_grad()
        output = model(data, adj_weights)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

    with open('./results/train.csv', 'a', newline='') as train_csv:
        writer = csv.writer(train_csv)
        writer.writerow([train_loss, 100. * correct / len(train_loader.dataset)])

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, adj_weights in test_loader:
            data, target, adj_weights = data.to(device), target.to(device), adj_weights.to(device)
            output = model(data, adj_weights)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if args.dry_run:
                break

    test_loss /= len(test_loader.dataset)

    with open('./results/test.csv', 'a', newline='') as test_csv:
        writer = csv.writer(test_csv)
        writer.writerow([test_loss, 100. * correct / len(test_loader.dataset)])

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def read_adj_matrix(path: Path | str) -> torch.Tensor:
    adj_matrix = torch.load(path)
    adj_matrix = adj_matrix - torch.eye(len(adj_matrix))
    adj_matrix = torch.argwhere(adj_matrix).T
    return adj_matrix


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example using RNN')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--step-size', type=int, default=20, metavar='N',
                        help='period of learning rate decay (default: 20)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='learning rate step gamma (default: 0.7)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--mps', action="store_true", default=False,
                        help="enables MPS training")
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='for Saving the current Model')
    args = parser.parse_args()

    if args.cuda and not args.mps:
        device = "cuda"
    elif args.mps and not args.cuda:
        device = "mps"
    else:
        device = "cpu"

    device = torch.device(device)

    generator = torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    """
    epochs_dataset = EpochsDataset('./working_data/004')
    train_dataset, test_dataset = torch.utils.data.random_split(epochs_dataset, [0.8, 0.2], generator)
    """

    adj_dataset = AdjDataset('./working_data/004', 'working_data_pli/004')
    train_dataset, test_dataset = torch.utils.data.random_split(adj_dataset, [0.8, 0.2], generator)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, collate_fn=adj_collate_fn, generator=generator, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, shuffle=True, collate_fn=adj_collate_fn, generator=generator, **kwargs)

    #model = GNN(device).to(device)
    model = GNN2(device).to(device)
    #model = EEGNet().to(device)
    #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-2)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_rnn.pt")


if __name__ == '__main__':
    main()


