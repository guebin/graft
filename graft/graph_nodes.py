import matplotlib as mpl

def set_nodes(gt_graph, num_nodes):
    """
    Add nodes to a graph_tool graph.

    Parameters:
        gt_graph (graph_tool.Graph): Input graph in graph_tool format.
        num_nodes (int): Number of nodes to add to the graph.

    Returns:
        vertices (list): List of vertices.
        gt_graph (graph_tool.Graph): Modified graph with added nodes.
    """
    vertices = [gt_graph.add_vertex() for _ in range(num_nodes)]
    return vertices, gt_graph

def map_vertex_names(gt_graph, vertices, node_names):
    """
    Map node names to vertices in a graph_tool graph.

    Parameters:
        gt_graph (graph_tool.Graph): Input graph in graph_tool format.
        vertices (list): List of vertices.
        node_names (list): List of node names.

    Returns:
        v_text_prop (graph_tool.VertexPropertyMap): Vertex property map with node names.
        gt_graph (graph_tool.Graph): Modified graph with vertex names.
    """
    if node_names is None:
        return None, vertices, gt_graph
    else: 
        v_text = gt_graph.new_vertex_property("string")
        for v, name in zip(vertices, node_names):
            v_text[v] = name
        return v_text, vertices, gt_graph

def map_vertex_color(gt_graph, vertices, node_colors):
    """
    Map y values to vertex colors for a graph_tool graph.

    Parameters:
        gt_graph (graph_tool.Graph): Input graph in graph_tool format.
        num_nodes (int): Number of nodes in the graph.
        y (torch.Tensor): Tensor with values to map to colors.

    Returns:
        v_color (graph_tool.VertexPropertyMap): Vertex property map with colors.
    """

    if node_colors is None:
        return None, vertices, gt_graph
    else: 
        y = list(node_colors)
        v_color = gt_graph.new_vertex_property("vector<double>")

        # Continuous values or too many unique values
        if any(isinstance(element, float) for element in y) or len(set(y))>10:
            y_min, y_max = min(y), max(y)
            colormap = mpl.cm.get_cmap('spring')
            for color, value in enumerate(v_color,y):
                normalized_value = (value.item() - y_min) / (y_max - y_min)  # Normalize between [0, 1]
                rgba = list(colormap(normalized_value))  # Convert to RGBA color
                color = rgba
        else: # Categorical or discrete values
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

            def hex_to_rgb_normalized(hex_color):
                rgb = mpl.colors.hex2color(hex_color)  # Gives RGB values between 0 and 1
                return [float(val) for val in rgb]

            ggplot_colors_rgb = [hex_to_rgb_normalized(color) for color in ggplot_colors]
            unique_y = list(set(y))
            unique_y.sort()
            y_to_color = {int(val): ggplot_colors_rgb[val] for val in enumerate(unique_y)}
            for idx, value in enumerate(y):
                rgb = y_to_color[int(value.item())]
                v_color[vertices[idx]] = rgb

        return v_color, vertices, gt_graph

def map_vertex_size(gt_graph, vertices, node_sizes):
    """
    Map y values to vertex sizes for a graph_tool graph.

    Parameters:
        gt_graph (graph_tool.Graph): Input graph in graph_tool format.
        vertices (list): List of vertex indices.
        node_size (torch.Tensor): Tensor with values to map to sizes.

    Returns:
        v_size (graph_tool.VertexPropertyMap): Vertex property map with sizes.
    """

    if node_sizes is None:
        return None, vertices, gt_graph
    else: 
        y = torch.tensor(node_sizes)
        v_size = gt_graph.new_vertex_property("double")

        y_min, y_max = y.min().item(), y.max().item()
        min_size, max_size = 10, 50  # Define the range of vertex sizes
        for idx, value in enumerate(y):
            normalized_value = (value.item() - y_min) / (y_max - y_min)  # Normalize between [0, 1]
            size = min_size + normalized_value * (max_size - min_size)  # Map to size range
            v_size[vertices[idx]] = size

        return v_size, vertices, gt_graph