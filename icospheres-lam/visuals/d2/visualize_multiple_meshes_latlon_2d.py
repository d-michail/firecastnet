import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ..utils import (
    configure_axis_labels,
    crop_latlon_map,
    face_wraps_around_latlon,
    get_colors, 
    get_edge_color,
    save_figure_if_path_provided,
    setup_cartopy_map,
    configure_cartopy_grid,
    to_lat_lon,
    toggle_figure_title
)

def visualize_multiple_meshes_latlon_2d(meshes, colors=None, labels=None, alpha=0.7, figsize=(15, 10),
                                       show_wireframe=True, title=None, show_grid_labels=True, show_title=True, show_labels=True, crop_bounds=None, save_path=None):
    """
    Visualize multiple meshes in 2D latitude-longitude projection with world map background.
    
    Parameters
    ----------
    meshes : list of dict
        List of mesh dictionaries, each containing:
        - 'vertices': np.ndarray of shape (N, 3) with 3D vertex coordinates
        - 'faces': np.ndarray of shape (M, 3) with face indices
    colors : list of str, optional
        List of colors for each mesh. If None, uses a default color cycle.
    labels : list of str, optional
        List of labels for each mesh. If None, uses mesh['label'] if available or defaults to 'Mesh N'.
    alpha : float, optional
        Transparency of the surfaces, by default 0.7
    figsize : tuple, optional
        Figure size as (width, height), by default (15, 10)
    show_wireframe : bool, optional
        Whether to show the wireframe edges, by default True
    title : str, optional
        Custom title for the plot. If None, generates automatic title.
    show_grid_labels : bool, optional
        Whether to show grid labels on the map, by default True
    show_title : bool, optional
        Whether to show the plot title, by default True
    show_labels : bool, optional
        Whether to show axis labels, by default True
    crop_bounds : dict, optional
        Dictionary containing crop parameters with keys 'lat' and 'lon'.
        Each value can be either a list/array [min, max] or a single number N for [-N, N].
        Example: {'lat': [20, 60], 'lon': [-10, 40]} or {'lat': 45, 'lon': 90}
        If None, shows global extent.
    save_path : str, optional
        Path where to save the figure (relative or absolute). If None, figure is not saved.
        File extension determines format (e.g., .png, .pdf, .svg, .jpg), by default None.
        
    Returns
    -------
    fig, ax : matplotlib figure and axis objects
        The created figure and cartopy axis for further customization
    """
    
    # Default colors if none provided
    if colors is None:
        colors = get_colors(len(meshes))
    
    # Create figure and axis with cartopy projection
    fig, ax, projection = setup_cartopy_map(None, figsize=figsize)
    
    # Configure grid labels
    if not show_grid_labels:
        configure_cartopy_grid(ax, show_labels=False)
    
    legend_elements = []
    all_vertices_latlon = []  # Store all vertices for extent calculation
    
    # Process each mesh
    for i, mesh in enumerate(meshes):
        vertices_3d = mesh['vertices']
        faces = mesh['faces']

        label = labels[i] if labels else f'Mesh {i+1}'
        color = colors[i]
        
        # Convert 3D vertices to lat-lon
        vertices_latlon = to_lat_lon(vertices_3d)
        all_vertices_latlon.append(vertices_latlon)  # Store for extent calculation
        
        # Plot vertices
        ax.scatter(vertices_latlon[:, 1], vertices_latlon[:, 0], 
                  color=color, s=3, alpha=0.8, transform=projection)
        
        # Plot faces
        for face in faces:
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
                    linewidth=0.2 if show_wireframe else 0,
                    alpha=alpha,
                    transform=projection)
        
        # Add to legend
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=8, 
                                    label=f'{label}'))
    
    # Set extent based on crop_bounds
    if crop_bounds is not None:
        # Apply cropping if specified
        crop_latlon_map(ax, crop_bounds)
    else:
        # No crop_bounds specified, show whole world
        ax.set_global()
    
    # Add title
    if title is None:
        title = f'Multiple Meshes Visualization - World Map ({len(meshes)} meshes)'
    
    # Get current figure and set title
    fig = plt.gcf()
    toggle_figure_title(fig, title, show_title, fontsize=14, fontweight='bold')
    
    # Configure axis labels (note: for Cartopy axes, this mainly affects title display)
    configure_axis_labels(ax, show_labels=show_labels, title=None)  # Title handled by toggle_figure_title

    # Add legend only if show_labels is True
    if show_labels:
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    
    # Show the plot
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure_if_path_provided(fig, save_path)
    
    plt.show()
    
    return fig, ax