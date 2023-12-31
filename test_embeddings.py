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
from torch_geometric.transforms import RandomLinkSplit
import numpy as np
from sklearn.metrics import roc_auc_score

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

# ---------------------------------------------- LINK PREDICTION TASK: ERIC's IMPLEMENTATION   --------------------------------------------

def train_test_split_graph(data, is_undirected = True):
    """
    Splits a Graph into a test and train set randomly to 80-20. The test split is balanced with negative edges sampled from random vertex pairs that have no edges between them. 
    While removing edges randomly, it makes sure that no vertex is isolated.
    :param data: a torch geometrics graph to be split
    :return: the train-test split as torch geometrics graphs
    """
    
    transform = RandomLinkSplit(num_val=0, num_test=0.2, is_undirected = is_undirected, add_negative_train_samples=False)
    train_data, _, test_data = transform(data)
    return train_data, test_data

def predict_link(u, v, embeddings):
    """
    Computes the normalized probability for an existing link between two nodes u and v based on the input
    embeddings.
    :param u: a node in the graph
    :param v: a node in the graph
    :param embeddings: trained embeddings
    :return: sigmoid normalized probability for the existence of a link
    """
    embedding1 = embeddings[u]
    embedding2 = embeddings[v]
    
    # Compute inner product (dot product)
    dot_product = np.dot(embedding1, embedding2)

    # Normalize by sigmoid function
    link_probability = 1/(1 + np.exp(-dot_product))
    return link_probability

def link_predictions(embeddings, edges, y_true):
    """
    Computes the ROC-AUC score for a given set of test edges based on the trained embeddings.
    :param embeddings: a models trained embeddings
    :param edges: test edges
    :param y_true: the labels for edges (1=true, 0=false)
    :return: the ROC-AUC score from predictions
    """
    predictions = []
    for edge in edges:
        predictions.append(predict_link(edge[0], edge[1], embeddings))
    return roc_auc_score(y_true, predictions) 

def k_fold_cross_validation_link_prediction(embeddings, edges, y_true, k=5):
    kf = KFold(n_splits=k, shuffle=True)
    auc_scores = []

    for train_index, test_index in kf.split(edges):
        # Splitting the data into training and test sets
        _, test_edges = edges[train_index], edges[test_index]
        _, y_test = y_true[train_index], y_true[test_index]

        # Evaluating the model on the test set
        score = link_predictions(embeddings, test_edges, y_test)
        auc_scores.append(score)

    return np.mean(auc_scores) 


# ---------------------------------------------- LINK PREDICTION TASK: JACK's IMPLEMENTATION + Auxiliary methods  --------------------------------------------

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
    print('preparation of data done')

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
    
    print('method gets called')

    # Get lists of nodes that form an edge
    edge_index = data.edge_index
    # make edges a list with 2 elements per entry 
    edges = list(zip(edge_index[0].numpy(), edge_index[1].numpy()))

    # Generate a set with all possible combinations of existing edges
    all_possible_pairs = set(combinations(range(data.num_nodes), 2))    

    # Set edges_exist based on whether the graph is directed or not
    if directed: 
        # For directed graphs, keep the edge direction
        edges_exist = list(set(edges))
    else:
        # For undirected graphs, sort the nodes in each edge
        edges_exist = list(set(map(lambda e: tuple(sorted(e)), edges)))

    # Subtract from all possible pairs, the ones that already exist
    non_edges = list(all_possible_pairs - set(edges_exist))

    return edges, non_edges