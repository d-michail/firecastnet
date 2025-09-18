import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from ..utils import (
    configure_axis_labels,
    configure_cartopy_grid,
    crop_latlon_map,
    face_wraps_around_latlon,
    get_edge_color,
    save_figure_if_path_provided,
    setup_cartopy_map,
    to_lat_lon,
    toggle_figure_title,
    order
)
from matplotlib.patches import Patch

def visualize_icosphere_latlon_2d(icospheres, refinement_order=0, color=None, show_wireframe=True, alpha=0.7, 
                                polygon_vertices=None, polygon_centroid=None,
                                show_face_numbers=False, show_grid_labels=True, show_title=True, show_labels=True, crop_bounds=None, save_path=None):
    """
    Visualize an icosphere in 2D latitude-longitude space with a world map background.
    
    Parameters
    ----------
    icospheres : dict
        Dictionary containing icosphere data
    refinement_order : int, optional
        The order of the icosphere to visualize, by default 0
    color : str, optional
        Color for the icosphere surfaces and vertices. If None, uses default 'cyan' for faces and 'blue' for vertices.
    show_wireframe : bool, optional
        Whether to show the wireframe of the icosphere, by default True
    alpha : float, optional
        Transparency of the surface, by default 0.7
    polygon_vertices : np.ndarray, optional
        Array of shape (N, 3) representing the polygon vertices in 3D
    polygon_centroid : np.ndarray, optional
        Array of shape (3,) representing the polygon centroid in 3D
    show_face_numbers : bool, optional
        Whether to display face indices in the triangles, by default False
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
    save_path : str, optional
        Path where to save the figure (relative or absolute). If None, figure is not saved.
        File extension determines format (e.g., .png, .pdf, .svg, .jpg), by default None.
        
    Returns
    -------
    None
        Displays a matplotlib 2D visualization with world map background
    """
    
    # Get vertices and faces for the specified order
    vertices_3d = icospheres[order(refinement_order, "vertices")]
    faces = icospheres[order(refinement_order, "faces")]

    # Convert 3D vertices to lat-lon
    vertices_latlon = to_lat_lon(vertices_3d)
    
    # Set default colors if none provided
    if color is None:
        vertex_color = 'b'  # Default blue for vertices
        face_color = 'cyan'  # Default cyan for faces
    else:
        vertex_color = color
        face_color = color
    
    # Create figure and axis with projection
    fig, ax, projection = setup_cartopy_map(None, figsize=(12, 8))
    
    # Configure grid labels
    if not show_grid_labels:
        configure_cartopy_grid(ax, show_labels=False)
    
    # Plot vertices
    ax.scatter(vertices_latlon[:, 1], vertices_latlon[:, 0], 
              color=vertex_color, s=10, alpha=0.6, transform=projection)
        
    # Prepare for face numbers if needed
    face_centers = []
    face_indices = []
    
    # Plot all faces
    for i, face in enumerate(faces):
        if face_wraps_around_latlon(vertices_latlon[face]):
            # Skip faces that wrap around the longitude boundary
            continue        

        # Get lat-lon coordinates for this face's vertices
        triangle_latlon = vertices_latlon[face]
        
        # Extract longitude and latitude 
        triangle_lons = triangle_latlon[:, 1]
        triangle_lats = triangle_latlon[:, 0]
        
        # Determine edge color
        edge_color = get_edge_color(face_color)
            
        # Plot the triangle
        ax.fill(triangle_lons, triangle_lats,
                facecolor=face_color, 
                edgecolor=edge_color if show_wireframe else None,
                linewidth=0.5 if show_wireframe else 0,
                alpha=alpha,
                transform=projection)
        
        # Calculate and store the face center and index if we're showing face numbers
        if show_face_numbers:
            # Calculate the center of the triangle in lat-lon space
            center_lon = np.mean(triangle_lons)
            center_lat = np.mean(triangle_lats)
            face_centers.append((center_lon, center_lat))
            face_indices.append(i)
    
    # Display face numbers if requested
    if show_face_numbers:
        for (center_lon, center_lat), face_idx in zip(face_centers, face_indices):
            ax.text(center_lon, center_lat, 
                   str(face_idx), 
                   color='black', fontsize=8, 
                   ha='center', va='center', fontweight='bold')
    
    # Initialize legend elements
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=vertex_color, markersize=8, label='Icosphere Vertices'),
        Patch(facecolor=face_color, edgecolor=edge_color if show_wireframe else None, alpha=alpha, label='Icosphere Faces')
    ]
        
    # Plot polygon if provided 
    if polygon_vertices is not None:
        if not isinstance(polygon_vertices, np.ndarray):
            polygon_vertices = np.array(polygon_vertices)
        polygon_lons = polygon_vertices[:, 1]
        polygon_lats = polygon_vertices[:, 0]
        
        # Plot polygon vertices
        ax.scatter(polygon_lons, polygon_lats, 
                  color='r', s=20, alpha=1.0)
        
        # Connect polygon vertices to form edges
        for i in range(len(polygon_lons)):
            j = (i + 1) % len(polygon_lons)
            ax.plot([polygon_lons[i], polygon_lons[j]],
                   [polygon_lats[i], polygon_lats[j]],
                   color='r', linewidth=2)
        
        # Fill the polygon area
        ax.fill(polygon_lons, polygon_lats,
               facecolor='red', edgecolor='darkred',
               linewidth=1.0, alpha=0.3)
        
        # Add to legend
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='r', 
                                    markersize=4, label='Polygon Vertices'))
        legend_elements.append(Patch(facecolor='red', edgecolor='darkred', alpha=0.3, label='Polygon Face'))
    
    # Plot centroid if provided
    if polygon_centroid is not None:
        # Convert the 3D centroid to lat-lon
        centroid_latlon = to_lat_lon(polygon_centroid.reshape(1, 3))
        centroid_lon = centroid_latlon[0, 1]
        centroid_lat = centroid_latlon[0, 0]
        
        ax.scatter(centroid_lon, centroid_lat, 
                  color='g', s=50, marker='*', alpha=1.0)
        
        legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='g', 
                                    markersize=10, label='Polygon Centroid'))
    
    # Add title
    title_text = f'Icosphere (Order {refinement_order}) in Lat-Lon Projection'
    if polygon_vertices is not None:
        title_text += ' with Spherical Polygon'
    if show_face_numbers:
        title_text += ' (with face indices)'
    
    # Get current figure and set title
    fig = plt.gcf()
    toggle_figure_title(fig, title_text, show_title, fontsize=14, fontweight='bold')
    
    # Configure axis labels (note: for Cartopy axes, this mainly affects title display)
    configure_axis_labels(ax, show_labels=show_labels, title=None)  # Title handled by toggle_figure_title
    
    # Apply cropping if specified
    if crop_bounds is not None:
        crop_latlon_map(ax, crop_bounds)
    else:
        # Set extent to show the whole world
        ax.set_global()
    
    # Add legend
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Show the plot
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure_if_path_provided(fig, save_path)
    
    plt.show()
