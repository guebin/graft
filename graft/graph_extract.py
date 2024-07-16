import torch

def extract_graph_components(graph):
    """
    Extract important components of a PyTorch Geometric graph. 
    This function is tailored for directed, weighted graphs.

    Parameters:
        graph (torch_geometric.data.Data): Input graph in PyTorch Geometric format.

    Returns:
        links (torch.Tensor): Tensor representing graph edges.
        weights (torch.Tensor): Tensor representing edge weights.
        num_nodes (int): Number of nodes in the graph.
        x (torch.Tensor, optional): Node feature matrix.
        y (torch.Tensor, optional): Node labels or target values.
    """
    # Extract edges (links) directly
    if graph.is_undirected():
        unique_edges = set(tuple(sorted(edge)) for edge in graph.edge_index .t().tolist())
        links = torch.tensor(list(unique_edges)).t().long()
    else:
        links = graph.edge_index      
    # Extract weights
    weights = torch.tensor(graph.edge_attr) if (hasattr(graph, 'edge_attr') and graph.edge_attr is not None) else torch.ones(links.size(1))

    # Extract number of nodes
    num_nodes = graph.num_nodes

    # Extract node features (x) and labels/targets (y)
    x = torch.tensor(graph.x) if (hasattr(graph, 'x') and graph.x is not None) else None
    y = torch.tensor(graph.y) if (hasattr(graph, 'y') and graph.y is not None) else None

    return links, weights, num_nodes, x, y
