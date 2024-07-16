import torch

# def extract_graph_components_uduw(graph):
#     """
#     Extract important components of a PyTorch Geometric graph. 
#     This function is tailored for undirected, unweighted graphs.

#     Parameters:
#         graph (torch_geometric.data.Data): Input graph in PyTorch Geometric format.

#     Returns:
#         links (torch.Tensor): Tensor representing graph edges.
#         num_nodes (int): Number of nodes in the graph.
#         x (torch.Tensor, optional): Node feature matrix.
#         y (torch.Tensor, optional): Node labels or target values.
#     """
#     # Extract unique edges (links)
#     unique_edges = set(tuple(sorted(edge)) for edge in graph.edge_index .t().tolist()) 
#     links = torch.tensor(list(unique_edges)).t().long()

#     # Extract number of nodes
#     num_nodes = graph.num_nodes

#     # Extract node features (x) and labels/targets (y)
#     x = torch.tensor(graph.x) if (hasattr(graph, 'x') and graph.x is not None) else None
#     y = torch.tensor(graph.y) if (hasattr(graph, 'y') and graph.y is not None) else None

#     return links, num_nodes, x, y

# def extract_graph_components_udw(graph):
#     """
#     Extract important components of a PyTorch Geometric graph. 
#     This function is tailored for undirected, weighted graphs.

#     Parameters:
#         graph (torch_geometric.data.Data): Input graph in PyTorch Geometric format.

#     Returns:
#         links (torch.Tensor): Tensor representing graph edges.
#         num_nodes (int): Number of nodes in the graph.
#         x (torch.Tensor, optional): Node feature matrix.
#         y (torch.Tensor, optional): Node labels or target values.
#     """
#     # Extract unique edges (links)
#     unique_edges_dict = {tuple(e.tolist()): w.item() for e, w in zip(graph.edge_index.sort(axis=0)[0].t(), graph.edge_attr)}
#     links = torch.stack([torch.tensor(k) for k in unique_edges_dict]).t().long()

#     # Extract weights 
#     weights = torch.tensor([v for v in unique_edges_dict.values()]).float()

#     # Extract number of nodes
#     num_nodes = graph.num_nodes

#     # Extract node features (x) and labels/targets (y)
#     x = torch.tensor(graph.x) if (hasattr(graph, 'x') and graph.x is not None) else None
#     y = torch.tensor(graph.y) if (hasattr(graph, 'y') and graph.y is not None) else None

#     return links, weights, num_nodes, x, y
# def extract_graph_components_duw(graph):
#     """
#     Extract important components of a PyTorch Geometric graph. 
#     This function is tailored for directed, unweighted graphs.

#     Parameters:
#         graph (torch_geometric.data.Data): Input graph in PyTorch Geometric format.

#     Returns:
#         links (torch.Tensor): Tensor representing graph edges.
#         num_nodes (int): Number of nodes in the graph.
#         x (torch.Tensor, optional): Node feature matrix.
#         y (torch.Tensor, optional): Node labels or target values.
#     """
#     # Extract edges (links) directly
#     links = graph.edge_index

#     # Extract number of nodes
#     num_nodes = graph.num_nodes

#     # Extract node features (x) and labels/targets (y)
#     x = torch.tensor(graph.x) if (hasattr(graph, 'x') and graph.x is not None) else None
#     y = torch.tensor(graph.y) if (hasattr(graph, 'y') and graph.y is not None) else None

#     return links, num_nodes, x, y

def _extract_graph_components(graph):
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
        links = graph.edge_index
    else:
        unique_edges = set(tuple(sorted(edge)) for edge in graph.edge_index .t().tolist())
        links = torch.tensor(list(unique_edges)).t().long()       
    # Extract weights
    weights = torch.tensor(graph.edge_attr) if (hasattr(graph, 'edge_attr') and graph.edge_attr is not None) else torch.ones(links.size(1))

    # Extract number of nodes
    num_nodes = graph.num_nodes

    # Extract node features (x) and labels/targets (y)
    x = torch.tensor(graph.x) if (hasattr(graph, 'x') and graph.x is not None) else None
    y = torch.tensor(graph.y) if (hasattr(graph, 'y') and graph.y is not None) else None

    return links, weights, num_nodes, x, y
