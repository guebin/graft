import graph_tool.all as gt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import matplotlib.cm as cm

def set_links_unweighted(gt_graph,links):
    gt_graph.add_edge_list(links.t().tolist())
    return gt_graph

def set_links_weighted(gt_graph,vertices,links,weights,edge_weight_text_format,edge_weight_width_scale,arrow_size_scale):
    e_weight = gt_graph.new_edge_property("string")
    e_pen_width = gt_graph.new_edge_property("double")
    e_arrow_size = gt_graph.new_edge_property("double")
    for idx, (start, end) in enumerate(links.t().tolist()):
        e = gt_graph.add_edge(vertices[start], vertices[end])
        e_weight[e] = format(weights[idx].item(),edge_weight_text_format)
        e_pen_width[e] = (weights/weights.max())[idx].item() * edge_weight_width_scale
        e_arrow_size[e] = (weights / weights.max())[idx].item() * arrow_size_scale
    return e_weight, e_pen_width, e_arrow_size, gt_graph