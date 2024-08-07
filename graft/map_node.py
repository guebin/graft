import matplotlib as mpl
import numpy as np

def map_node_names(gt_graph, draw_options, node_names):
    """
    Map node names to vertices in a graph_tool graph.

    Parameters:
        gt_graph (graph_tool.Graph): Input graph in graph_tool format.
        draw_options (dict): Dictionary of drawing options for the graph_draw function.
        node_names (list): List of node names.

    Returns:
        gt_graph (graph_tool.Graph): Modified graph with vertex names.
        draw_options (dict): Updated drawing options with vertex names.
    """
    if node_names is None:
        return gt_graph, draw_options
    else:
        v_text = gt_graph.new_vertex_property("string")
        for idx, name in enumerate(node_names):
            v_text[idx] = name
        draw_options['vertex_text'] = v_text
        return gt_graph, draw_options
    
def map_node_colors(gt_graph, draw_options, node_colors, alpha=1.0):
    """
    Map node colors to vertices in a graph_tool graph.

    Parameters:
        gt_graph (graph_tool.Graph): Input graph in graph_tool format.
        draw_options (dict): Dictionary of drawing options for the graph_draw function.
        node_colors (list): List of node colors.
        alpha (float, optional): Alpha value for node colors. Default is 1.0.

    Returns:
        gt_graph (graph_tool.Graph): Modified graph with vertex colors.
        draw_options (dict): Updated drawing options with vertex colors.
    """
    if node_colors is None:
        return gt_graph, draw_options
    else:
        y = list(np.array(node_colors))
        v_color = gt_graph.new_vertex_property("vector<double>")

        # Continuous values or too many unique values
        if any(isinstance(element, float) for element in y) or len(set(y)) > 10:
            y_min, y_max = min(y), max(y)
            colormap = mpl.cm.get_cmap('spring')
            for idx, value in enumerate(y):
                normalized_value = (value - y_min) / (y_max - y_min)  # Normalize between [0, 1]
                rgba = list(colormap(normalized_value))  # Get RGBA color
                rgba[-1] = alpha  # Set the alpha value
                v_color[idx] = rgba
        else:  # Categorical or discrete values
            colors_dict = {
                1: ['#F8766D'],
                2: ['#F8766D', '#00BFC4'],
                3: ['#F8766D', '#00BA38', '#619CFF'],
                4: ['#F8766D', '#7CAE00', '#00BFC4', '#C77CFF'],
                5: ['#F8766D', '#A3A500', '#00BF7D', '#00B0F6', '#E76BF3'],
                6: ['#F8766D', '#B79F00', '#00BA38', '#00BFC4', '#619CFF', '#F564E3'],
                7: ['#F8766D', '#C49A00', '#53B400', '#00C094', '#00B6EB', '#A58AFF', '#FB61D7'],
                8: ['#F8766D', '#CD9600', '#7CAE00', '#00BE67', '#00BFC4', '#00A9FF', '#C77CFF', '#FF61CC'],
                9: ['#F8766D', '#D39200', '#93AA00', '#00BA38', '#00C19F', '#00B9E3', '#619CFF', '#DB72FB', '#FF61C3'],
                10: ['#F8766D', '#A3A500', '#39B600', '#00BF7D', '#00BFC4', '#00B0F6', '#9590FF', '#E76BF3', '#FF62BC', '#D89000']
            }
            ggplot_colors = colors_dict[len(set(y))]

            def hex_to_rgb_normalized(hex_color, alpha=alpha):
                rgb = mpl.colors.hex2color(hex_color)  # Gives RGB values between 0 and 1
                rgba = list(rgb) + [alpha]
                return [float(val) for val in rgba]

            ggplot_colors_rgb = [hex_to_rgb_normalized(color) for color in ggplot_colors]
            unique_y = list(set(y))
            unique_y.sort()
            y_to_color = {int(val): ggplot_colors_rgb[val] for val in unique_y}
            for idx, value in enumerate(y):
                rgb = y_to_color[int(value)]
                v_color[idx] = rgb
        draw_options['vertex_fill_color'] = v_color
        return gt_graph, draw_options

def map_node_sizes(gt_graph, draw_options, node_sizes):
    """
    Map node sizes to vertices in a graph_tool graph.

    Parameters:
        gt_graph (graph_tool.Graph): Input graph in graph_tool format.
        draw_options (dict): Dictionary of drawing options for the graph_draw function.
        node_sizes (list): List of node sizes.

    Returns:
        gt_graph (graph_tool.Graph): Modified graph with vertex sizes.
        draw_options (dict): Updated drawing options with vertex sizes.
    """
    if node_sizes is None:
        return gt_graph, draw_options
    else:
        y = list(np.array(node_sizes))
        v_size = gt_graph.new_vertex_property("double")

        y_min, y_max = min(y), max(y)
        min_size, max_size = 10, 50  # Define the range of vertex sizes
        for idx, value in enumerate(y):
            normalized_value = (value - y_min) / (y_max - y_min)  # Normalize between [0, 1]
            size = min_size + normalized_value * (max_size - min_size)  # Map to size range
            v_size[idx] = size
        draw_options['vertex_size'] = v_size
        return gt_graph, draw_options