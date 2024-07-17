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

    # 3. Create a graph_tool graph.
    if graph.is_undirected():
        gt_graph = gt.Graph(directed=False)
        for _ in range(graph.num_nodes):
            gt_graph.add_vertex()
        gt_graph.data = graph
    else:
        gt_graph = gt.Graph(directed=True)    
        for _ in range(graph.num_nodes):
            gt_graph.add_vertex() 
        gt_graph.data = graph

    # 5. Map names, colors and sizes.
    gt_graph, draw_options = map_vertex_size(gt_graph, draw_options, node_sizes)
    gt_graph, draw_options = map_vertex_color(gt_graph, draw_options, node_colors)
    gt_graph, draw_options = map_vertex_names(gt_graph, draw_options, node_names)
    gt_graph, draw_options = map_edge_pen_widths(gt_graph, draw_options)
    #gt_graph, draw_options = map_edge_weight_text(gt_graph, draw_options)

    # # 6. Set links.
    # e_weight, e_pen_width, gt_graph = set_links(
    #     gt_graph,
    #     vertices,
    #     links,
    #     weights,
    #     edge_weight_text_format='.2f',
    #     edge_weight_width_scale=5.0,
    # )
    
    # # 7. Set draw_options.
    # if node_names is not None:
    #     draw_options['vertex_text'] = v_text
    # if node_colors is not None:
    #     draw_options['vertex_fill_color'] = v_color  
    # if node_sizes is not None:
    #     draw_options['vertex_size'] = v_size  
    # if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
    #     draw_options['edge_text'] = e_weight
    #     draw_options['edge_pen_width'] = e_pen_width  

    # 8. Perform graph layout using sfdf_layout and draw the graph using graph_draw.
    draw_options['pos'] = gt.sfdp_layout(gt_graph, **layout_options)    
    return gt_graph, draw_options     
   
def plot(
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
        layout_options=layout_options,
        draw_options=draw_options,
    )
    gt.graph_draw(gt_graph, **draw_options)