import numpy as np
import matplotlib.pyplot as plt
from dgl import DGLGraph
from matplotlib.lines import Line2D

def to_np(x):
    return x.detach().cpu().numpy()

def visualize_graph_matplotlib(graph: DGLGraph, title="Graph Visualization", node_size=20, edge_width=1, 
                              show_edges=True, alpha=0.7, node_color='blue', edge_color='green'):
    """
    Visualize a DGL graph in 3D using Matplotlib.
    
    Parameters
    ----------
    graph : DGLGraph
        The graph to visualize
    title : str, optional
        Title of the plot, by default "Graph Visualization"
    node_size : int, optional
        Size of nodes in the visualization, by default 20
    edge_width : float, optional
        Width of edges in the visualization, by default 0.5
    show_edges : bool, optional
        Whether to show edges, by default True
    alpha : float, optional
        Transparency of the visualization elements, by default 0.7
    node_color : str, optional
        Color of the nodes, by default 'blue'
    edge_color : str, optional
        Color of the edges, by default 'green'
        
    Returns
    -------
    None
        Displays a matplotlib 3D visualization
    """
    # Check if graph has position information
    if 'pos' in graph.ndata or 'x' in graph.ndata:
        # Get node positions
        if 'pos' in graph.ndata:
            if isinstance(graph.ndata['pos'], dict):
                # For heterogeneous graphs
                node_types = list(graph.ndata['pos'].keys())
                all_pos = []
                for ntype in node_types:
                    all_pos.append(to_np(graph.ndata['pos'][ntype]))
                pos = np.vstack(all_pos)
            else:
                # For homogeneous graphs
                pos = to_np(graph.ndata['pos'])
        else:
            pos = to_np(graph.ndata['x'])
    else:
        print("Graph doesn't have position information ('pos' or 'x' in ndata)")
        return
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot nodes
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], 
               color=node_color, s=node_size, alpha=alpha)
    
    # Plot edges if requested
    if show_edges:
        # Get edges
        src, dst = graph.edges()
        src = to_np(src)
        dst = to_np(dst)

        # Draw edges
        for i in range(len(src)):
            source = pos[src[i]]
            target = pos[dst[i]]
            ax.plot([source[0], target[0]], 
                    [source[1], target[1]], 
                    [source[2], target[2]], 
                    color=edge_color, linewidth=edge_width, alpha=alpha*0.7)
    
    # Draw a transparent sphere as background
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    
    # Calculate radius based on node positions
    radius = np.max(np.linalg.norm(pos, axis=1)) * 1.02
    
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot the sphere with a slight transparency
    ax.plot_surface(x, y, z, color='skyblue', alpha=0.1, linewidth=0)
    
    # Draw latitude/longitude grid lines
    # Meridians (lines of longitude)
    for phi in np.linspace(0, 2*np.pi, 12, endpoint=False):
        x = radius * np.cos(phi) * np.sin(np.linspace(0, np.pi, 50))
        y = radius * np.sin(phi) * np.sin(np.linspace(0, np.pi, 50))
        z = radius * np.cos(np.linspace(0, np.pi, 50))
        ax.plot(x, y, z, 'gray', alpha=0.2, linewidth=0.5)
        
    # Parallels (lines of latitude)
    for theta in np.linspace(0.2, np.pi-0.2, 6):
        x = radius * np.cos(np.linspace(0, 2*np.pi, 50)) * np.sin(theta)
        y = radius * np.sin(np.linspace(0, 2*np.pi, 50)) * np.sin(theta)
        z = radius * np.cos(theta) * np.ones(50)
        ax.plot(x, y, z, 'gray', alpha=0.2, linewidth=0.5)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Set axis limits
    max_range = np.array([pos[:, 0].max() - pos[:, 0].min(),
                         pos[:, 1].max() - pos[:, 1].min(),
                         pos[:, 2].max() - pos[:, 2].min()]).max() / 2.0
    
    mid_x = (pos[:, 0].max() + pos[:, 0].min()) * 0.5
    mid_y = (pos[:, 1].max() + pos[:, 1].min()) * 0.5
    mid_z = (pos[:, 2].max() + pos[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add title and labels
    plt.title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add a legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=node_color, 
               markersize=8, label='Nodes'),
    ]
    
    if show_edges:
        legend_elements.append(
            Line2D([0], [0], color=edge_color, linewidth=2, label='Edges')
        )
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Remove background grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)
    
    # Show the plot
    plt.tight_layout()
    plt.show()