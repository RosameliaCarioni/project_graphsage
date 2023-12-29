import torch
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
import time
from sklearn.metrics import f1_score 

def visualize_information_graph(dataset):
    data = dataset[0]
    print(f'Dataset: {dataset}')
    print('-------------------')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of nodes: {data.x.shape[0]}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    # Print information about the graph
    print(f'\nGraph:')
    print('------')
    print(f'Training nodes: {sum(data.train_mask).item()}')
    print(f'Evaluation nodes: {sum(data.val_mask).item()}')
    print(f'Test nodes: {sum(data.test_mask).item()}')
    print(f'Edges are directed: {data.is_directed()}')
    print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Graph has loops: {data.has_self_loops()}')


def visualize_information_loader(loader):
    # Print each subgraph
    for i, subgraph in enumerate(loader):
        print(f'Subgraph {i}: {subgraph}')