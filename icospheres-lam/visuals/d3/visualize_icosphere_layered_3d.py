import numpy as np
import matplotlib.pyplot as plt
from ..utils import (
    get_colors, 
    create_3d_mesh_collection, 
    extract_icosphere_orders,
    configure_axis_labels,
    configure_3d_grid,
    save_figure_if_path_provided,
    toggle_figure_title,
    order
)


def visualize_icosphere_layered_3d(icosphere_layers, max_columns=3, show_wireframe=True, alpha=0.7,
                                  rotation=(0, 0), figsize=(15, 10), colors=None,
                                  zoom=None, title="Icosphere Refinement Layers",
                                  show_labels=True, show_grid=True, show_title=True, save_path=None):
    """
    Visualize multiple icosphere refinement layers/orders in a grid layout with 3D subplots.
    
    Parameters
    ----------
    icosphere_layers : dict
        Dictionary containing icosphere data with keys like 'order_0_vertices', 'order_0_faces', etc.
    max_columns : int, optional
        Maximum number of columns in the subplot grid, by default 3
    show_wireframe : bool, optional
        Whether to show the wireframe of the icospheres, by default True
    alpha : float, optional
        Transparency of the surfaces, by default 0.7
    rotation : tuple, optional
        Rotation angles as (elevation, azimuth) in degrees, by default (30, 45)
    figsize : tuple, optional
        Figure size as (width, height), by default (15, 10)
    colors : list of str, optional
        List of colors for each layer. If None, uses a default color cycle.
    zoom : list of float, optional
        List of zoom factors for each subfigure. Values > 1.0 zoom in, < 1.0 zoom out.
        If None, uses 1.0 for all subfigures. If shorter than number of layers, 
        repeats the last value for remaining subfigures.
    title : str, optional
        Main title for the entire figure, by default "Icosphere Refinement Layers"
    show_labels : bool, optional
        Whether to show axis labels and subplot titles, by default True
    show_grid : bool, optional
        Whether to show background grid on axes, by default True
    show_title : bool, optional
        Whether to show the figure title, by default True
    save_path : str, optional
        Path where to save the figure (relative or absolute). If None, figure is not saved.
        File extension determines format (e.g., .png, .pdf, .svg, .jpg), by default None.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object for further customization
    """
    # Extract available orders from the dictionary
    available_orders = extract_icosphere_orders(icosphere_layers)
    num_layers = len(available_orders)
    
    if num_layers == 0:
        print("No icosphere layers found in the provided data.")
        return None
    
    # Calculate grid dimensions
    num_cols = min(max_columns, num_layers)
    num_rows = (num_layers + num_cols - 1) // num_cols  # Ceiling division
    
    # Default colors if none provided
    if colors is None:
        colors = get_colors(num_layers)
        
    # Handle zoom factors
    if zoom is None:
        zoom = [1.0] * num_layers
    elif len(zoom) < num_layers:
        # Extend zoom list by repeating the last value
        last_zoom = zoom[-1] if zoom else 1.0
        zoom.extend([last_zoom] * (num_layers - len(zoom)))
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Set figure title
    toggle_figure_title(fig, title, show_title, fontsize=16, fontweight='bold')
    
    # Store all vertices for consistent axis scaling
    all_vertices = []
    for o in available_orders:
        vertices = icosphere_layers[order(o, "vertices")]
        all_vertices.append(vertices)
    
    # Calculate global axis limits
    if all_vertices:
        all_points = np.vstack(all_vertices)
        max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                             all_points[:, 1].max() - all_points[:, 1].min(),
                             all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0
        
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
    
    # Create subplots for each layer
    for i, o in enumerate(available_orders):        
        # Create 3D subplot
        ax = fig.add_subplot(num_rows, num_cols, i + 1, projection='3d')
        
        # Get vertices and faces for this order
        vertices = icosphere_layers[order(o, "vertices")]
        faces = icosphere_layers[order(o, "faces")]
        color = colors[i]
        
        # Plot vertices
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                  color=color, s=5, alpha=0.8)
        
        # Create and add 3D mesh collection
        collection = create_3d_mesh_collection(vertices, faces, color, show_wireframe, alpha, edge_alpha=0.7)
        if collection:
            ax.add_collection3d(collection)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Apply individual zoom for this subplot
        zoom_factor = zoom[i]
        zoomed_range = max_range / zoom_factor
        
        # Calculate zoomed axis limits
        zoomed_xlim = (mid_x - zoomed_range, mid_x + zoomed_range)
        zoomed_ylim = (mid_y - zoomed_range, mid_y + zoomed_range)
        zoomed_zlim = (mid_z - zoomed_range, mid_z + zoomed_range)
        
        # Set individual axis limits for this subplot
        ax.set_xlim(zoomed_xlim)
        ax.set_ylim(zoomed_ylim)
        ax.set_zlim(zoomed_zlim)
        
        # Apply rotation
        ax.view_init(elev=rotation[0], azim=rotation[1])
        
        # Configure labels and grid using new utility functions
        configure_axis_labels(ax, show_labels=show_labels, title=f'Order {o}')
        configure_3d_grid(ax, show_grid=show_grid)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, hspace=0.3)  # More room for main title and spacing between rows
    
    # Save figure if path provided
    save_figure_if_path_provided(fig, save_path)
    
    plt.show()
    
    return fig

