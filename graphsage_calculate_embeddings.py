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
from torch_geometric.loader import LinkNeighborLoader

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
    normalize
):
    # Sampling from neighbourhood
    train_loader = LinkNeighborLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        neg_sampling_ratio=0.25,  # generating 50% of negative nodes
        num_neighbors=[neighborhood_1, neighborhood_2],
    )

    if torch.backends.mps.is_available() and False:
        device = torch.device("mps")
        x = torch.ones(1, device=device)
    else:
        device = torch.device("cpu")

    # Create model
    model = GraphSAGE(
        in_channels=number_features,
        hidden_channels=hidden_layer,  # TODO: not sure
        out_channels=embedding_dimension,
        num_layers=2,
        aggr=aggregator,
        act=activation_function,
        dropout=dropout_rate,
        act_first=activation_before_normalization,
        bias=bias,
        normalize = normalize,

    ).to(device)

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
    """_summary_

    Args:
        model: _description_
        device: _description_
        train_loader: _description_
        optimizer: _description_
        number_nodes: _description_

    Returns:
        average loss 
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