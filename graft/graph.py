from .graph_extract import *
from .graph_nodes import * 
from .graph_links import * 
import graph_tool.all as gt
    
def setup_graph_draw(
        graph,
        node_names=None,
        node_colors=None,
        node_sizes=None,
        layout_options=None,
        draw_options=None,
    ):
    """
    Visualize a directed and weighted graph.

    Parameters:
        graph (torch_geometric.data.Data): Input graph in PyTorch Geometric format.
        node_names (list, optional): List of node names. Default is None.
        node_colors (list, optional): List of node colors. Default is None.
        node_sizes (list, optional): List of node sizes. Default is None.
        edge_weight_text (bool, optional): Whether to display edge weights as text. Default is True.
        edge_weight_width (bool, optional): Whether to adjust edge widths based on weights. Default is True.
        edge_weight_text_format (str, optional): Format string for edge weight text. Default is ".2f".
        edge_weight_width_scale (float, optional): Scale factor for edge width based on weights. Default is 1.0.
        layout_options (dict, optional): Dictionary of layout options for the layout algorithm. Default is None.
        draw_options (dict, optional): Dictionary of drawing options for the graph drawing. Default is None.

    Returns:
        gt_graph (graph_tool.Graph): Prepared graph_tool graph.
        draw_options (dict): Options for the graph_draw function.
    """
    # Default layout options
    default_layout_options = {
        'vweight': None,
        'eweight': None,
        'pin': None,
        'C': 0.2,
        'K': None,
        'p': 2.0,
        'theta': 0.6,
        'max_level': 15,
        'r': 1.0,
        'kc': 10,
        'groups': None,
        'gamma': 0.1,
        'mu': 2.0,
        'kappa': 1.0,
        'rmap': None,
        'R': 1,
        'init_step': None,
        'cooling_step': 0.95,
        'adaptive_cooling': True,
        'epsilon': 1e-2,
        'max_iter': 0,
        'pos': None,
        'multilevel': None,
        'coarse_method': "hybrid",
        'mivs_thres': 0.9,
        'ec_thres': 0.75,
        'weighted_coarse': False,
        'verbose': False
    }

    # Update layout options with user-provided values
    if layout_options:
        default_layout_options.update(layout_options)
    layout_options = default_layout_options

    # Default draw options
    default_draw_options = {
        'pos': None,
        'vprops': None,
        'eprops': None,
        'vorder': None,
        'eorder': None,
        'nodesfirst': False,
        'output_size': (300, 300),
        'fit_view': True,
        'fit_view_ink': None,
        'adjust_aspect': True,
        'ink_scale': 1,
        'inline': True,
        'inline_scale': 2,
        'mplfig': None,
        'yflip': True,
        'output': None,
        'fmt': 'auto',
        'bg_color': None,
        'antialias': None,
    }

    # Update draw options with user-provided values
    if draw_options:
        default_draw_options.update(draw_options)
    draw_options = default_draw_options

    # Extract_graph_components.
    links, weights, num_nodes  = extract_graph_components(graph)

    # 3. Create a graph_tool graph.
    if graph.is_undirected():
        gt_graph = gt.Graph(directed=False)
    else:
        gt_graph = gt.Graph(directed=True)    
    if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
        edge_weight_text=True,
        edge_weight_width=True
    else:
        edge_weight_text=False,
        edge_weight_width=False,

    # 4. Set nodes.
    vertices, gt_graph = set_nodes(gt_graph, num_nodes)

    # 5. Map names, colors and sizes.
    v_size, vertices, gt_graph = map_vertex_size(gt_graph, vertices, node_sizes)
    v_color, vertices, gt_graph = map_vertex_color(gt_graph, vertices, node_colors)
    v_text, vertices, gt_graph = map_vertex_names(gt_graph, vertices, node_names)

    # 6. Set links.
    e_weight, e_pen_width, gt_graph = set_links(
        gt_graph,
        vertices,
        links,
        weights,
        edge_weight_text_format='.2f',
        edge_weight_width_scale=1.0,
    )
    
    # 7. Set draw_options.
    if node_names is not None:
        draw_options['vertex_text'] = v_text
    if node_colors is not None:
        draw_options['vertex_fill_color'] = v_color  
    if node_sizes is not None:
        draw_options['vertex_size'] = v_size  
    if edge_weight_text: 
        draw_options['edge_text'] = e_weight  
    if edge_weight_width: 
        draw_options['edge_pen_width'] = e_pen_width  

    # 8. Perform graph layout using sfdf_layout and draw the graph using graph_draw.
    draw_options['pos'] = gt.sfdp_layout(gt_graph, **layout_options)    
    return gt_graph, draw_options     
   
def plot(
        graph,
        node_names=None,
        node_colors=None,
        node_sizes=None,
        edge_weight_text=True,
        edge_weight_width=True,
        edge_weight_text_format=".2f",
        edge_weight_width_scale=1.0,
        layout_options=None,
        draw_options=None,
    ):
    """
    Visualize a directed and weighted graph.

    Parameters:
        graph (torch_geometric.data.Data): Input graph in PyTorch Geometric format.
        node_names (list, optional): List of node names. Default is None.
        node_colors (list, optional): List of node colors. Default is None.
        node_sizes (list, optional): List of node sizes. Default is None.
        edge_weight_text (bool, optional): Whether to display edge weights as text. Default is True.
        edge_weight_width (bool, optional): Whether to adjust edge widths based on weights. Default is True.
        edge_weight_text_format (str, optional): Format string for edge weight text. Default is ".2f".
        edge_weight_width_scale (float, optional): Scale factor for edge width based on weights. Default is 1.0.
        layout_options (dict, optional): Dictionary of layout options for the layout algorithm. Default is None.
        draw_options (dict, optional): Dictionary of drawing options for the graph drawing. Default is None.

    Returns:
        None
    """
    gt_graph, draw_options = setup_graph_draw(
        graph,
        node_names=node_names,
        node_colors=node_colors,
        node_sizes=node_sizes,
        edge_weight_text=edge_weight_text,
        edge_weight_width=edge_weight_width,
        edge_weight_text_format=edge_weight_text_format,
        edge_weight_width_scale=edge_weight_width_scale,
        layout_options=layout_options,
        draw_options=draw_options,
    )
    gt.graph_draw(gt_graph, **draw_options)