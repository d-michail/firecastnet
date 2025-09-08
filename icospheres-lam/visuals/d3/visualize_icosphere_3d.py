from matplotlib.patches import Patch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from ..utils import (
    create_3d_mesh_collection, 
    configure_axis_labels,
    configure_3d_grid,
    save_figure_if_path_provided,
    setup_3d_axis_equal_aspect,
    toggle_figure_title,
    order
)

def visualize_icosphere_3d(icosphere, refinement_order=0, show_wireframe=True, alpha=0.7, 
                          polygon_vertices=None, polygon_centroid=None, show_labels=True, show_grid=True, show_title=True, save_path=None):
    """
    Visualize an icosphere using Matplotlib, optionally with a spherical polygon and its centroid.

    Parameters
    ----------
    icosphere : dict
        Dictionary containing icosphere data
    refinement_order : int, optional
        The refinement_order of the icosphere to visualize, by default 0
    show_wireframe : bool, optional
        Whether to show the wireframe of the icosphere, by default True
    alpha : float, optional
        Transparency of the surface, by default 0.7
    polygon_vertices : np.ndarray, optional
        Array of shape (N, 3) representing the polygon vertices
    polygon_centroid : np.ndarray, optional
        Array of shape (3,) representing the polygon centroid
    show_labels : bool, optional
        Whether to show axis labels and title, by default True
    show_grid : bool, optional
        Whether to show the 3D grid, by default True
    show_title : bool, optional
        Whether to show the figure title, by default True
    save_path : str, optional
        Path where to save the figure (relative or absolute). If None, figure is not saved.
        File extension determines format (e.g., .png, .pdf, .svg, .jpg), by default None.

    Returns
    -------
    None
        Displays a matplotlib 3D visualization
    """

    # Get vertices and faces for the specified order
    vertices = icosphere[order(refinement_order, "vertices")]
    faces = icosphere[order(refinement_order, "faces")]

    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
              color='b', s=10, alpha=0.6)

    # Create and add 3D mesh collection
    collection = create_3d_mesh_collection(vertices, faces, 'cyan', show_wireframe, alpha)
    if collection:
        ax.add_collection3d(collection)

    # Initialize legend elements
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=8, label='Icosphere Vertices'),
        Patch(facecolor='cyan', edgecolor='darkblue', alpha=alpha, label='Icosphere Faces')
    ]

    # Plot polygon if provided (place it outside the sphere)
    if polygon_vertices is not None:
        # Scale the polygon vertices to place outside the sphere (e.g., 1.2 times the radius)
        scale_factor = 1.05
        scaled_polygon_vertices = polygon_vertices * scale_factor

        # Plot scaled polygon vertices
        ax.scatter(scaled_polygon_vertices[:, 0], scaled_polygon_vertices[:, 1], scaled_polygon_vertices[:, 2], 
                  color='r', s=30, alpha=1.0)

        # Connect scaled polygon vertices to form edges
        for i in range(len(scaled_polygon_vertices)):
            j = (i + 1) % len(scaled_polygon_vertices)
            ax.plot([scaled_polygon_vertices[i, 0], scaled_polygon_vertices[j, 0]],
                   [scaled_polygon_vertices[i, 1], scaled_polygon_vertices[j, 1]],
                   [scaled_polygon_vertices[i, 2], scaled_polygon_vertices[j, 2]],
                   'r-', linewidth=2)

        # Add scaled polygon face
        poly_collection = Poly3DCollection([scaled_polygon_vertices], 
                                         facecolors='red', 
                                         edgecolors='darkred',
                                         linewidths=1.0,
                                         alpha=0.3)
        ax.add_collection3d(poly_collection)

        # Add to legend
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='r', 
                                    markersize=8, label='Polygon Vertices'))
        legend_elements.append(Patch(facecolor='red', edgecolor='darkred', alpha=0.3, label='Polygon Face'))

    # Plot centroid if provided (place it outside the sphere)
    if polygon_centroid is not None and polygon_vertices is not None:
        # Scale the centroid to match the polygon vertices
        scaled_centroid = polygon_centroid * scale_factor

        ax.scatter(scaled_centroid[0], scaled_centroid[1], scaled_centroid[2], 
                  color='g', s=50, marker='*', alpha=1.0)

        legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='g', 
                                    markersize=10, label='Polygon Centroid'))

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Set axis limits accounting for the scaled polygon
    all_points = vertices
    if polygon_vertices is not None:
        all_points = np.vstack((all_points, scaled_polygon_vertices))
    if polygon_centroid is not None:
        all_points = np.vstack((all_points, scaled_centroid.reshape(1, 3)))

    setup_3d_axis_equal_aspect(ax, all_points)

    # Configure labels and title
    title_text = f'Icosphere (Order {refinement_order})'
    if polygon_vertices is not None:
        title_text += ' with Spherical Polygon'
    configure_axis_labels(ax, show_labels=show_labels, title=title_text if show_labels else None,
                       fontsize_labels=12, fontsize_title=14)
    
    # Set figure title
    toggle_figure_title(fig, title_text, show_title, fontsize=16)
    
    # Configure grid
    configure_3d_grid(ax, show_grid=show_grid)

    # Add legend
    ax.legend(handles=legend_elements, loc='upper right')

    # Show the plot
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure_if_path_provided(fig, save_path)
    
    plt.show()
