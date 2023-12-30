import torch
from torch_geometric.data import Data
import pandas as pd
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import graphsage_experiments
import torch.nn.functional as F

def add_masks(data, train_size, val_size, test_size):
    """Add masks to the data object 

    Args:
        data: torch_geometric.data
        train_size: ratio 
        val_size: ratio
        test_size: ratio

    Returns:
        train mask, validation mask, test mask 
    """
    # Ensure the sizes sum up to the number of nodes
    assert train_size + val_size + test_size <= data.num_nodes

    # Shuffle node indices
    node_indices = torch.randperm(data.num_nodes)

    # Split indices for train, val, test
    train_indices = node_indices[:train_size]
    val_indices = node_indices[train_size:train_size + val_size]
    test_indices = node_indices[train_size + val_size:]

    # Initialize masks
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    # Assign masks
    data.train_mask[train_indices] = True
    data.val_mask[val_indices] = True
    data.test_mask[test_indices] = True

    return data


def read_dataset_arizona_university(nodes_path, edges_path, groups_path, group_edges_path):
    """Read in the data from the csv files and transform it into networkx object 

    Args:
        nodes_path: path to the csv file 
        edges_path: path to the csv file 
        groups_path: path to the csv file 
        group_edges_path: path to the csv 
            
    Returns:
        graph, labels or classes to the nodes 
    """
    nodes_id = pd.read_csv(nodes_path, header=None, names=['id'])
    #groups_id = pd.read_csv(groups_path, header=None, names=['group'])
    edges = pd.read_csv(edges_path, header=None, names=['id_1', 'id_2'])
    user_group_membership = pd.read_csv(group_edges_path, header=None, names=['id', 'group'])

    # Create a graph
    graph = nx.Graph()

    # Add nodes to the graph
    #G_BC.add_nodes_from(nodes_id['id'])
    for node_id in nodes_id['id']:
        graph.add_node(node_id, id=node_id)
    
    # Add edges (friendships) to the graph
    graph.add_edges_from(edges[['id_1', 'id_2']].values)

    # Create a dictionary to store groups for each ID
    group_dict = {}

    # Populate the group_dict
    for _, row in user_group_membership.iterrows():
        user_id = row['id']
        group_id = row['group']

        # Check if the user_id is already in the dictionary
        if user_id in group_dict:
            group_dict[user_id].append(group_id)
        else:
            group_dict[user_id] = [group_id]

    # Add group labels to the nodes
    for user_id, groups in group_dict.items():
        nx.set_node_attributes(graph, {user_id: groups}, 'y') # 'group belonging'
    
    # Find and preprocess labels for the graph
    labels = []
    c = 0 
    for n in graph.nodes:
        l = graph.nodes[n].get('y')
        labels.append(l)

    mlb = MultiLabelBinarizer()
    preprocessed_labels = mlb.fit_transform(labels)

    return graph, preprocessed_labels

