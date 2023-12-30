import time
from tqdm import tqdm

import numpy as np 
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, train_test_split
from sklearn.metrics import roc_auc_score
from itertools import combinations


# ---------------------------------------------- NODE CLASSIFICATION TASK --------------------------------------------

@torch.no_grad()
def test_node_classification_one_class(embedding_matrix, y, n_folds=5):
    """ 5-fold classification using one-vs-rest logistic regression
    Args:
        embedding_matrix: source embeddings of a graph
        y: the labels for each node
        n_folds: number of folds for cross-validation
    Returns: 
        accuracy, f1 macro score, f1 micro score
    """

    model = LogisticRegression(multi_class="ovr")
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=123)
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
        n_folds: number of folds for cross-validation. Defaults to 5
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


# ---------------------------------------------- LINK PREDICTION TASK --------------------------------------------

@torch.no_grad()
def test_link_prediction_k_fold_validation(embedding, edges, non_edges, k=5):
    """5 fold cross validation for link prediction 

    Args:
        embedding: numpy.ndarray with the embedding representations for each node in graph
        edges: list of edges containig 2 nodes per entry
        non_edges: list of edges that don't exist in the graph and could be added
        k: number of folds in k-fold cross validation.  Defaults to 5.

    Returns:
        average ROC AUC
    """

   # PREPARE DATA 
    # Extract embeddings for the given edges and non-edges
    emb_edges = np.array([embedding[edge[0]] * embedding[edge[1]] for edge in edges])
    emb_non_edges = np.array([embedding[non_edge[0]] * embedding[non_edge[1]] for non_edge in non_edges])

    # Label the edges as 1 and non-edges as 0
    labels = np.concatenate([np.ones(len(edges)), np.zeros(len(non_edges))])

    # Combine edge and non-edge embeddings
    X = np.concatenate([emb_edges, emb_non_edges])

    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # List to store ROC_AUC scores for each fold
    roc_auc_scores = []

    # PERFORM K-FOLD CROSS VALIDATION
    for train_index, test_index in kf.split(X):
        # Split data into training and testing sets for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Normalize the inner product by the sigmoid function
        inner_product = np.sum(X_test, axis=1)
        normalized_similarity = 1 / (1 + np.exp(-inner_product))

        # Compute and store the ROC-AUC score
        roc_auc = roc_auc_score(y_test, normalized_similarity)
        roc_auc_scores.append(roc_auc)

    # Calculate the average ROC AUC score across all folds
    avg_roc_auc = np.mean(roc_auc_scores)

    return avg_roc_auc

def test_link_prediction(embedding, edges, non_edges):
    """Result for link prediction given embedding matrix of nodes 

    Args:
        embedding: numpy.ndarray with the embedding representations for each node in graph
        edges: list of edges containig 2 nodes per entry
        non_edges: list of edges that don't exist in the graph and could be added

    Returns:
        ROC AUC
    """

    # Extract embeddings for the given edges and non-edges
    emb_edges = np.array([embedding[edge[0]] * embedding[edge[1]] for edge in edges])
    emb_non_edges = np.array([embedding[non_edge[0]] * embedding[non_edge[1]] for non_edge in non_edges])

    # Label the edges as 1 and non-edges as 0
    labels = np.concatenate([np.ones(len(edges)), np.zeros(len(non_edges))])

    # Combine edge and non-edge embeddings
    X = np.concatenate([emb_edges, emb_non_edges])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5, random_state=42)

    # Normalize the inner product by the sigmoid function
    inner_product = np.sum(X_test, axis=1)
    normalized_similarity = 1 / (1 + np.exp(-inner_product))

    # Compute ROC-AUC score
    roc_auc = roc_auc_score(y_test, normalized_similarity)

    return roc_auc*100

def get_edges_and_non_edges_as_lists(data, directed): 
    """From a torch_geometric.datasets obtain the list of edges and possible edges that don't exist 

    Args:
        data: torch_geometric.datasets
        directed: boolean that says if the graph is directed or not 

    Returns:
        edges and non_edges in lists of tuples where each element is one node  
    """

    # Get lists of nodes that form an edge
    edge_index = data.edge_index

    # make edges a list with 2 elements per entry 
    edges = list(zip(edge_index[0].numpy(), edge_index[1].numpy()))

    # Generate a set with all possible combinations of existing edges
    all_possible_pairs = set(combinations(range(data.num_nodes), 2))

    # Get the unique pairs from edges - in case of undirected graphs 
    if not directed: 
        unique_edges = list(set(map(lambda e: tuple(sorted(e)), edges)))

    # Set unique_edges based on whether the graph is directed
    if directed: 
        # For directed graphs, keep the edge direction
        edges_exist = list(set(edges))
    else:
        # For undirected graphs, sort the nodes in each edge
        edges_exist = list(set(map(lambda e: tuple(sorted(e)), edges)))

    # Subtract from all possible pairs, the ones that already exist
    non_edges = list(all_possible_pairs - set(edges_exist))

    return edges, non_edges