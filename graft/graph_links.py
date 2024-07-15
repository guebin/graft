import graph_tool.all as gt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import matplotlib.cm as cm

def set_links_uduw(gt_graph,links):
    gt_graph.add_edge_list(links.t().tolist())
    return gt_graph

def set_links_udw(gt_graph,vertices,links,weights,edge_weight_text_format,edge_weight_width_scale):
    #vertices = {i: gt_graph.add_vertex() for i in range(num_nodes)}
    e_weight = gt_graph.new_edge_property("string")
    e_pen_width = gt_graph.new_edge_property("double")
    for idx, (start, end) in enumerate(links.t().tolist()):
        e = gt_graph.add_edge(vertices[start], vertices[end])
        e_weight[e] = format(weights[idx].item(),edge_weight_text_format)
        e_pen_width[e] = (weights/weights.max())[idx].item() * edge_weight_width_scale
    return e_weight, e_pen_width, gt_graph

def set_links_duw(gt_graph, links):
    """
    Set directed, unweighted links in the graph_tool graph.

    Parameters:
        gt_graph (graph_tool.Graph): Input graph in graph_tool format.
        links (torch.Tensor): Tensor representing graph edges.

    Returns:
        gt_graph (graph_tool.Graph): Updated graph with edges.
    """
    gt_graph.add_edge_list(links.t().tolist())
    return gt_graph

def set_links_dw(gt_graph, vertices, links, weights, edge_weight_text_format, edge_weight_width_scale):
    """
    Set directed, weighted links in the graph_tool graph.

    Parameters:
        gt_graph (graph_tool.Graph): Input graph in graph_tool format.
        vertices (list): List of vertex indices.
        links (torch.Tensor): Tensor representing graph edges.
        weights (torch.Tensor): Tensor representing edge weights.
        edge_weight_text_format (str): Format string for edge weight text.
        edge_weight_width_scale (float): Scaling factor for edge width based on weight.

    Returns:
        e_weight (graph_tool.EdgePropertyMap): Edge property map with weights as text.
        e_pen_width (graph_tool.EdgePropertyMap): Edge property map with edge widths.
        gt_graph (graph_tool.Graph): Updated graph with edges and properties.
    """
    e_weight = gt_graph.new_edge_property("string")
    e_pen_width = gt_graph.new_edge_property("double")
    for idx, (start, end) in enumerate(links.t().tolist()):
        e = gt_graph.add_edge(vertices[start], vertices[end])
        e_weight[e] = format(weights[idx].item(), edge_weight_text_format)
        e_pen_width[e] = (weights / weights.max())[idx].item() * edge_weight_width_scale
    return e_weight, e_pen_width, gt_graph    