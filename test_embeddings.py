import time
from tqdm import tqdm

import numpy as np 
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, train_test_split
from sklearn.metrics import roc_auc_score
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

# ---------------------------------------------- LINK PREDICTION TASK   --------------------------------------------

def train_test_split_graph(data, is_undirected = True):
    """
    Splits a Graph into a test and train set randomly to 50-50. The test split is balanced with negative edges sampled from random vertex pairs that have no edges between them. 
    While removing edges randomly, it makes sure that no vertex is isolated.
    :param data: a torch geometrics graph to be split
    :return: the train-test split as torch geometrics graphs
    """
    
    transform = RandomLinkSplit(num_val=0, num_test=0.5, is_undirected = is_undirected, add_negative_train_samples=False)
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