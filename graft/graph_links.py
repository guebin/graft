def set_edge_weight_text(
        gt_graph,
        vertices,
        links,
        weights,
        edge_weight_text_format
    ):
    """
    Set edge weights in the graph_tool graph.

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
    for idx, (start, end) in enumerate(links.t().tolist()):
        e = gt_graph.add_edge(vertices[start], vertices[end])
        e_weight_text[e] = format(weights[idx].item(), edge_weight_text_format)
    return e_weight_text, gt_grapha

def set_edge_pen_widths(
        gt_graph,
        vertices,
        links,
        weights,
        edge_weight_width_scale
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
    for idx, (start, end) in enumerate(links.t().tolist()):
        e = gt_graph.add_edge(vertices[start], vertices[end])
        e_pen_width[e] = (weights / weights.max())[idx].item() * edge_weight_width_scale
    return e_pen_width, gt_graph