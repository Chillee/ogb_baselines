import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm


from copy import deepcopy
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, APPNP
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import numpy as np

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torchvision import transforms
# from outcome_correlation import *
import glob
import os
import shutil

from logger import Logger
    
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.inProj = torch.nn.Linear(in_channels, hidden_channels)
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.linear = torch.nn.Linear(hidden_channels, out_channels)
        self.weights = torch.nn.Parameter(torch.randn((len(self.convs))))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.linear.reset_parameters()
        self.inProj.reset_parameters()
        torch.nn.init.normal_(self.weights)

    
    def pretrain(self, x, adj):
        out = []
        x = self.inProj(x)
        inp = x
        for i, conv in enumerate(self.convs):
            x = conv(x, adj)
            x = self.bns[i](x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + 0.2*inp
            out.append(x)
        sftmax = F.softmax(self.weights)
        for i in range(len(out)):
            out[i] = out[i] * sftmax[i]
        x = sum(out)
        x = self.linear(x)
        return x
    
    def forward(self, x, adj):
        x = self.pretrain(x, adj)
        return x.log_softmax(dim=-1)

        
def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, x, y, adj, split_idx, evaluator):
    model.eval()

    out = model(x, adj)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

    
        
            
def main():
    parser = argparse.ArgumentParser(description='gen_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--use_node_embedding', action='store_true')

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--runs', type=int, default=10)

    args = parser.parse_args()
    print(args)

    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv',transform=T.ToSparseTensor())
    
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)
    
    x = data.x
    if args.use_node_embedding:
        embedding = torch.load('embedding.pt', map_location=device)
        x = torch.cat([x, embedding], dim=-1)
        
    x = x.to(device)
    adj_t = data.adj_t.to(device)
    y_true = data.y.to(device)
    
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    valid_idx = split_idx['valid'].to(device)
    test_idx = split_idx['test'].to(device)

    model = GCN(x.size(-1), args.hidden_channels, dataset.num_classes, args.num_layers, args.dropout).cuda()
        
    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)
    
    idxs = torch.cat([train_idx])
    for run in range(args.runs):
        print(sum(p.numel() for p in model.parameters()))
        
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_valid = 0
        best_out = None
        
        import time
        begin = time.time()
        for epoch in range(1, args.epochs):
            model.train()
            optimizer.zero_grad()
            out = model(x, adj_t)[idxs]
            loss = F.nll_loss(out, y_true.squeeze(1)[idxs])
            result = test(model, x, y_true, adj_t, split_idx, evaluator)
            train_acc, valid_acc, test_acc = result
        
            print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
            logger.add_result(run, result)
            loss.backward()
            optimizer.step()
        logger.print_statistics(run)

    logger.print_statistics()




if __name__ == "__main__":
    main()
