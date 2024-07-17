from .map_vertex import * 
from .map_edge import * 
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
    Prepare a directed or undirected graph for visualization using graph_tool.

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
        'vweight': None, 'eweight': None, 'pin': None, 'C': 0.2, 'K': None, 'p': 2.0, 'theta': 0.6,
        'max_level': 15, 'r': 1.0, 'kc': 10, 'groups': None, 'gamma': 0.1, 'mu': 2.0, 'kappa': 1.0,
        'rmap': None, 'R': 1, 'init_step': None, 'cooling_step': 0.95, 'adaptive_cooling': True,
        'epsilon': 1e-2, 'max_iter': 0, 'pos': None, 'multilevel': None, 'coarse_method': "hybrid",
        'mivs_thres': 0.9, 'ec_thres': 0.75, 'weighted_coarse': False, 'verbose': False
    }
    # Update layout options with user-provided values
    if layout_options:
        default_layout_options.update(layout_options)
    layout_options = default_layout_options

    # Default draw options
    default_draw_options = {
        'pos': None, 'vprops': None, 'eprops': None, 'vorder': None, 'eorder': None,
        'nodesfirst': False, 'output_size': (300, 300), 'fit_view': True, 'fit_view_ink': None,
        'adjust_aspect': True, 'ink_scale': 1, 'inline': True, 'inline_scale': 2, 'mplfig': None,
        'yflip': True, 'output': None, 'fmt': 'auto', 'bg_color': None, 'antialias': None,
    }
    # Update draw options with user-provided values
    if draw_options:
        default_draw_options.update(draw_options)
    draw_options = default_draw_options

    # Create a graph_tool graph based on the directionality of the input graph
    gt_graph = gt.Graph(directed=not graph.is_undirected())
    for _ in range(graph.num_nodes):
        gt_graph.add_vertex()
    gt_graph.data = graph

    # Map node properties (size, color, names) and edge properties (pen widths)
    gt_graph, draw_options = map_vertex_size(gt_graph, draw_options, node_sizes)
    gt_graph, draw_options = map_vertex_color(gt_graph, draw_options, node_colors)
    gt_graph, draw_options = map_vertex_names(gt_graph, draw_options, node_names)
    gt_graph, draw_options = map_edge_pen_widths(gt_graph, draw_options)

    # Set the layout positions using the specified layout options
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
    Visualize a graph using graph_tool.

    Parameters:
        graph (torch_geometric.data.Data): Input graph in PyTorch Geometric format.
        node_names (list, optional): List of node names. Default is None.
        node_colors (list, optional): List of node colors. Default is None.
        node_sizes (list, optional): List of node sizes. Default is None.
        layout_options (dict, optional): Layout options for the layout algorithm. Default is None.
        draw_options (dict, optional): Drawing options for the graph_draw function. Default is None.

    Returns:
        None
    """
    # Prepare the graph and draw options
    gt_graph, draw_options = setup_graph_draw(
        graph,
        node_names=node_names,
        node_colors=node_colors,
        node_sizes=node_sizes,
        layout_options=layout_options,
        draw_options=draw_options,
    )
    # Draw the graph using graph_tool
    gt.graph_draw(gt_graph, **draw_options)ㄴ