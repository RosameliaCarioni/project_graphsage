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
from sklearn.linear_model import LogisticRegression
from torch_geometric.nn import GraphSAGE
from torch_geometric.loader import LinkNeighborLoader
from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import StratifiedKFold, KFold, cross_validate

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


# ---- NODE CLASSIFICATION TASK ----

@torch.no_grad()
def test_node_classification_one_class(embedding_matrix, y):
    """ 5-fold classification using one-vs-rest logistic regression
    Args:
        embedding_matrix: source embeddings of a graph
        y: the labels for each node
        n_folds: number of folds for cross-validation
    Returns: 
        accuracy, f1 macro score, f1 micro score
    """

    model = LogisticRegression(multi_class="ovr")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    eval_scores = {"acc": "accuracy", "f1_macro": "f1_macro", "f1_micro": "f1_micro"}

    results = cross_validate(model, embedding_matrix, y, cv=kf, scoring=eval_scores)

    acc, f1_macro, f1_micro = (
        results["test_acc"].mean(),
        results["test_f1_macro"].mean(),
        results["test_f1_micro"].mean(),
    )

    return acc, f1_macro, f1_micro

@torch.no_grad()
def test_node_classification_multi_class(embedding_matrix, y, n_folds=5):
    """ 5-fold multi-label classification using one-vs-rest logistic regression
    Args:
        embedding_matrix: source embeddings of a graph
        y: the labels for each node
        n_folds: number of folds for cross-validation
    Returns: 
        accuracy, f1 macro score, f1 micro score
    """
    model = LogisticRegression()
    ovr_model = OneVsRestClassifier(model)
    kf = KFold(n_splits=n_folds, shuffle=True)
    eval_scores = {'acc': 'accuracy', 'f1_macro': 'f1_macro', 'f1_micro': 'f1_micro'}
    results = cross_validate(ovr_model, embedding_matrix, y, cv=kf, scoring=eval_scores)
    acc, f1_macro, f1_micro = results['test_acc'].mean(), results['test_f1_macro'].mean(), results['test_f1_micro'].mean()
    return acc, f1_macro, f1_micro

@torch.no_grad()
def test_link_prediction(embedding_matrix, edge_label_index, y):

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    eval_scores = {"acc": "accuracy", "roc_auc": "roc_auc"}

    # TODO 
    return 0


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