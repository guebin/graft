def map_edge_weight_text(
        gt_graph,
        draw_options,
        edge_weight_text_format=".2f"
    ):
    """
    Map edge weights as text in the graph_tool graph.

    Parameters:
        gt_graph (graph_tool.Graph): Input graph in graph_tool format.
        vertices (list): List of vertex indices.
        links (torch.Tensor): Tensor representing graph edges.
        weights (torch.Tensor): Tensor representing edge weights.
        edge_weight_text_format (str): Format string for edge weight text.

    Returns:
        e_weight_text (graph_tool.EdgePropertyMap): Edge property map with weights as text.
        gt_graph (graph_tool.Graph): Updated graph with edges and properties.
    """
    e_weight_text = gt_graph.new_edge_property("string")
    vertices = list(gt_graph.vertices())

    # Extract edges (links) directly
    if gt_graph.is_directed():
        links = graph.edge_index
    else:
        unique_edges = set(tuple(sorted(edge)) for edge in graph.edge_index .t().tolist())
        links = torch.tensor(list(unique_edges)).t().long()
              
    # Extract weights
    weights = torch.tensor(graph.edge_attr) if (hasattr(graph, 'edge_attr') and graph.edge_attr is not None) else torch.ones(links.size(1))

    for idx, (start, end) in enumerate(links.t().tolist()):
        e = gt_graph.add_edge(vertices[start], vertices[end])
        e_weight_text[e] = format(weights[idx].item(), edge_weight_text_format)
    draw_options['edge_text'] = e_weight_text
    return gt_graph, draw_options

def map_edge_pen_widths(
        gt_graph,
        draw_options,
        edge_weight_width_scale=5
    ):
    """
    Set edge pen widths in the graph_tool graph.

    Parameters:
        gt_graph (graph_tool.Graph): Input graph in graph_tool format.
        vertices (list): List of vertex indices.
        links (torch.Tensor): Tensor representing graph edges.
        weights (torch.Tensor): Tensor representing edge weights.
        edge_weight_width_scale (float): Scaling factor for edge width based on weight.

    Returns:
        e_pen_width (graph_tool.EdgePropertyMap): Edge property map with edge widths.
        gt_graph (graph_tool.Graph): Updated graph with edges and properties.
    """
    e_pen_width = gt_graph.new_edge_property("double")
    vertices = list(gt_graph.vertices())
    # Extract edges (links) directly
    if gt_graph.is_directed():
        links = graph.edge_index
    else:
        unique_edges = set(tuple(sorted(edge)) for edge in graph.edge_index .t().tolist())
        links = torch.tensor(list(unique_edges)).t().long()
              
    # Extract weights
    weights = torch.tensor(graph.edge_attr) if (hasattr(graph, 'edge_attr') and graph.edge_attr is not None) else torch.ones(links.size(1))

    for idx, (start, end) in enumerate(links.t().tolist()):
        e = gt_graph.add_edge(vertices[start], vertices[end])
        e_pen_width[e] = (weights / weights.max())[idx].item() * edge_weight_width_scale
    draw_options['edge_pen_width'] = e_pen_width        
    return gt_graph, draw_options