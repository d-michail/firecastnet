import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ..utils import (
    configure_axis_labels,
    configure_cartopy_grid,
    crop_latlon_map,
    extract_icosphere_orders,
    face_wraps_around_latlon,
    get_colors,
    get_edge_color,
    save_figure_if_path_provided,
    setup_cartopy_map,
    to_lat_lon,
    toggle_figure_title,
    order
)

def visualize_icosphere_layered_latlon_2d(icosphere_layers, max_columns=3, show_wireframe=True, alpha=0.7,
                                         figsize=(20, 12), colors=None, labels=None,
                                         title="Icosphere Refinement Layers - Lat-Lon Projection", 
                                         show_grid_labels=True, show_title=True, show_labels=True, crop_bounds=None, save_path=None):
    """
    Visualize multiple icosphere refinement layers/orders in a grid layout with 2D lat-lon subplots.
    
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
    figsize : tuple, optional
        Figure size as (width, height), by default (20, 12)
    colors : list of str, optional
        List of colors for each layer. If None, uses a default color cycle.
    labels : list of str, optional
        List of labels for each subplot. If None, uses default format 'Order N'.
        If shorter than number of layers, uses default labels for remaining subfigures.
    title : str, optional
        Main title for the entire figure, by default "Icosphere Refinement Layers - Lat-Lon Projection"
    show_grid_labels : bool, optional
        Whether to show grid labels on edge subplots, by default True
    show_title : bool, optional
        Whether to show the figure title, by default True
    show_labels : bool, optional
        Whether to show axis labels, by default True
    crop_bounds : dict, optional
        Dictionary containing crop parameters with keys 'lat' and 'lon'.
        Each value can be either a list/array [min, max] or a single number N for [-N, N].
        Example: {'lat': [20, 60], 'lon': [-10, 40]} or {'lat': 45, 'lon': 90}
        Applied to all subplots.
    save_path : str, optional
        Path where to save the figure (relative or absolute). If None, figure is not saved.
        File extension determines format (e.g., .png, .pdf, .svg, .jpg), by default None.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object for further customization
    """
    import cartopy.crs as ccrs

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
    
    # Handle labels - extend with defaults if needed
    if labels is None:
        labels = [f'Order {o}' for o in available_orders]
    else:
        # Extend labels list with defaults if it's shorter than needed
        while len(labels) < num_layers:
            missing_order = available_orders[len(labels)]
            labels.append(f'Order {missing_order}')
    
    # Create figure with subplots using cartopy projection
    fig = plt.figure(figsize=figsize)
    
    # Set figure title
    toggle_figure_title(fig, title, show_title, fontsize=16, fontweight='bold')
    
    # Get projection for subplot creation
    projection = ccrs.PlateCarree()
    
    # Create subplots for each layer
    for i, o in enumerate(available_orders):
        
        # Create 2D subplot with cartopy projection
        _, ax, _ = setup_cartopy_map(fig, subplot_args=(num_rows, num_cols, i + 1))
        # Configure grid using utility function
        configure_cartopy_grid(ax, show_labels=False)
        
        # Get vertices and faces for this order
        vertices_3d = icosphere_layers[order(o, "vertices")]
        faces = icosphere_layers[order(o, "faces")]
        color = colors[i]
        
        # Convert 3D vertices to lat-lon
        vertices_latlon = to_lat_lon(vertices_3d)
        
        # Plot vertices
        ax.scatter(vertices_latlon[:, 1], vertices_latlon[:, 0], 
                  color=color, s=2, alpha=0.8, transform=projection)
        
        # Plot all faces
        for face_idx, face in enumerate(faces):
            # Check if face wraps around longitude boundary
            triangle_latlon = vertices_latlon[face]
            if face_wraps_around_latlon(triangle_latlon):
                # Skip faces that wrap around the longitude boundary
                continue
            
            # Extract longitude and latitude 
            triangle_lons = triangle_latlon[:, 1]
            triangle_lats = triangle_latlon[:, 0]
            
            # Determine edge color
            edge_color = get_edge_color(color)
            
            # Plot the triangle
            ax.fill(triangle_lons, triangle_lats,
                    facecolor=color, 
                    edgecolor=edge_color if show_wireframe else None,
                    linewidth=0.1 if show_wireframe else 0,
                    alpha=alpha,
                    transform=projection)
        
        # Apply cropping if specified, otherwise show the whole world
        if crop_bounds is not None:
            crop_latlon_map(ax, crop_bounds)
        else:
            # Set extent to show the whole world
            ax.set_global()
        
        # Add subtitle for this subplot using the custom label
        ax.set_title(f"Order {o}", fontsize=10, fontweight='bold', pad=15)
        
        # Configure axis labels (note: for Cartopy axes, this mainly affects title display)
        configure_axis_labels(ax, show_labels=show_labels, title=None)  # Title handled by ax.set_title above

        # Add gridlines with labels only for edge subplots to avoid clutter
        if show_grid_labels:
            if i % num_cols == 0:  # Left edge
                configure_cartopy_grid(ax, show_labels=True, position='left')
            elif i >= num_layers - num_cols:  # Bottom edge
                configure_cartopy_grid(ax, show_labels=True, position='bottom')
            else:
                configure_cartopy_grid(ax, show_labels=False)
        else:
            configure_cartopy_grid(ax, show_labels=False)
    
    # Add a legend for the colors if there's space
    if num_layers < num_rows * num_cols:
        # If there's space for a legend subplot
        legend_ax = fig.add_subplot(num_rows, num_cols, num_layers + 1)
        legend_ax.axis('off')
        
        # Create legend elements using custom labels
        legend_elements = []
        for i, o in enumerate(available_orders):
            color = colors[i]
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color, markersize=8, 
                      label=labels[i])
            )
        
        legend_ax.legend(handles=legend_elements, loc='center', 
                        title='Refinement Orders', title_fontsize=12)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.1)  # More room for main title and spacing
    
    # Save figure if path provided
    save_figure_if_path_provided(fig, save_path)
    
    plt.show()
    
    return fig
