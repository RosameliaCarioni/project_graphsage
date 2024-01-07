
def visualize_information_graph(dataset):
    """ Print information about a graph

    Args:
        dataset (torch_geometric.datasets)
    """
    data = dataset[0]

    print(f'Dataset: {dataset}')
    print('-------------------')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of nodes: {data.x.shape[0]}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    print(f'\nGraph:')
    print('------')
    print(f'Training nodes: {sum(data.train_mask).item()}')
    print(f'Evaluation nodes: {sum(data.val_mask).item()}')
    print(f'Test nodes: {sum(data.test_mask).item()}')
    print(f'Edges are directed: {data.is_directed()}')
    print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Graph has loops: {data.has_self_loops()}')


def visualize_information_loader(loader):
    """ Print information about loader

    Args:
        loader
    """
    # Print each subgraph
    for i, subgraph in enumerate(loader):
        print(f'Subgraph {i}: {subgraph}')