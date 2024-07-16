def set_links(
        gt_graph,
        vertices,
        links,
        weights,
        edge_weight_text_format,
        edge_weight_width_scale
    ):
    e_weight = gt_graph.new_edge_property("string")
    e_pen_width = gt_graph.new_edge_property("double")
    for idx, (start, end) in enumerate(links.t().tolist()):
        e = gt_graph.add_edge(vertices[start], vertices[end])
        e_weight[e] = format(weights[idx].item(),edge_weight_text_format)
        e_pen_width[e] = (weights/weights.max())[idx].item() * edge_weight_width_scale
    return e_weight, e_pen_width, gt_graph