# This work is inspired/based in the following work: 

# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_sage.py
# https://github.com/PacktPublishing/Hands-On-Graph-Neural-Networks-Using-Python/blob/main/Chapter08/chapter8.ipynb 
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_sage_unsup.py
# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html
# https://medium.com/@juyi.lin/neighborloader-introduction-ccb870cc7294

import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import LinkNeighborLoader

# ---------------------------------------------- GRAPH SAGE MODEL  --------------------------------------------

class GraphSAGE_local(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, aggr, normalize, activation_function, bias):
        """Initialize values and create layers of NN

        Args:
            in_channels
            hidden_channels 
            out_channels
            dropout
            aggr
            normalize
            activation_function
            bias
        """

        super().__init__()
        # as K = 2, we have 2 layers
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, out_channels = hidden_channels, project = activation_function, bias = bias, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, out_channels = out_channels, project = activation_function, bias = bias, aggr=aggr, normalize = normalize)
    

    def forward(self, matrix_nodes_features, edge_index):
        """Perform a forward pass in the NN

        Args:
            matrix_nodes_features: is a matrix from the data where row = nodes, columns = feature
            edge_index: This is a tensor that describes the connectivity of the graph. 
            Each column in this represents an edge. The first row contains the indices of the source nodes, and the second row contains the indices of the target nodes.

        Returns:
            output of model 
        """
    
        h = self.conv1(matrix_nodes_features, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout, training = self.training)

        h = self.conv2(h, edge_index)
        h = F.relu(h) 
        h = F.dropout(h,  p=self.dropout, training = self.training)
        return h

# ---------------------------------------------- EMBEDDING MATRIX  --------------------------------------------

def compute_embedding_matrix(
    data,
    number_features,
    number_nodes,
    batch_size,
    hidden_layer, 
    epochs, 
    neighborhood_1,
    neighborhood_2,
    embedding_dimension,
    learning_rate,
    dropout_rate,
    activation_function,
    aggregator,
    activation_before_normalization,
    bias,
    project,
    normalize
):
    """ Calculate the embedding matrix of a graph using GraphSage. 
    This method sets K=2 by using 2 layers in the network. 

    Args:
        data 
        number_features 
        number_nodes
        batch_size
        hidden_layer
        epochs
        neighborhood_1 
        neighborhood_2
        embedding_dimension (
        learning_rate 
        dropout_rate 
        activation_function 
        aggregator 
        activation_before_normalization 
        bias 
        project
        normalize 

    Returns:
        embedding matrix
    """
    # Sampling from neighbourhood
    train_loader = LinkNeighborLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        neg_sampling_ratio=20.0,
        num_neighbors=[neighborhood_1, neighborhood_2],
    )
    
    # This has False because I couldn't make it work on my macOS 14.1 - 14.2. 
    # For future work I will look further into it 
    if torch.backends.mps.is_available() and False:
        device = torch.device("mps")
        x = torch.ones(1, device=device)
    else:
        device = torch.device("cpu")


    # Create local model 
    model = GraphSAGE_local(
        in_channels=number_features,
        hidden_channels=hidden_layer, 
        out_channels=embedding_dimension,
        dropout=dropout_rate,
        aggr=aggregator,
        normalize = normalize,
        activation_function = activation_function,
        bias=bias,
    ).to(device)
    

    # Create model from libary - this was used to compare results 
    """
    model = GraphSAGE(
        in_channels=number_features,
        hidden_channels=hidden_layer, 
        out_channels=embedding_dimension,
        num_layers=2,
        aggr=aggregator,
        act=activation_function,
        dropout=dropout_rate,
        act_first=activation_before_normalization,
        bias=bias,
        normalize = normalize,
        project = project
    ).to(device) 
    """
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    times = []

    for epoch in tqdm(range(epochs + 1), desc='Training Progress'):
        
        start = time.time()
        total_loss = train(model, device, train_loader, optimizer, number_nodes)

        model.eval()
        embedding_matrix = model(data.x, data.edge_index).to(device)

        print(f"Epoch: {epoch:03d}, Total loss: {total_loss:.4f}, time_taken: {time.time() - start}")

        times.append(time.time() - start)
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

    return embedding_matrix

def train(model, device, train_loader, optimizer, number_nodes):
    """train model

    Args:
        model
        device
        train_loader
        optimizer
        number_nodes

    Returns:
        average loss of training process
    """    
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        h = model(batch.x, batch.edge_index)
        h_src = h[batch.edge_label_index[0]]
        h_dst = h[batch.edge_label_index[1]]
        pred = (h_src * h_dst).sum(dim=-1)

        loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pred.size(0)

    return total_loss / number_nodes