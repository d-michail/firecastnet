
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ..utils import (
    get_colors, 
    create_3d_mesh_collection, 
    get_edge_color, 
    save_figure_if_path_provided,
    setup_3d_axis_equal_aspect,
    configure_axis_labels,
    configure_3d_grid,
    toggle_figure_title
)


def visualize_multiple_meshes_3d(meshes, colors=None, alpha=0.7, figsize=(12, 10), 
                                show_labels=True, show_grid=True, show_title=True, save_path=None):
    """
    Visualize multiple meshes in 3D using matplotlib with unique colors and wireframes.
    
    Parameters
    ----------
    meshes : list of dict
        List of mesh dictionaries, each containing:
        - 'vertices': np.ndarray of shape (N, 3) with vertex coordinates
        - 'faces': np.ndarray of shape (M, 3) with face indices
        - 'label': str, optional label for the mesh (for legend)
    colors : list of str, optional
        List of colors for each mesh. If None, uses a default color cycle.
    alpha : float, optional
        Transparency of the surfaces, by default 0.7
    figsize : tuple, optional
        Figure size as (width, height), by default (12, 10)
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
    fig, ax : matplotlib figure and axis objects
        The created figure and 3D axis for further customization
    """
    
    # Default colors if none provided
    if colors is None:
        colors = get_colors(len(meshes))
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Store all vertices for axis scaling
    all_vertices = []
    legend_elements = []
    
    # Process each mesh
    for i, mesh in enumerate(meshes):
        vertices = mesh['vertices']
        faces = mesh['faces']
        label = mesh.get('label', f'Mesh {i+1}')
        color = colors[i]
        
        # Store vertices for axis scaling
        all_vertices.append(vertices)
        
        # Plot vertices
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                  color=color, s=10, alpha=0.8)
        
        # Create and add 3D mesh collection
        collection = create_3d_mesh_collection(vertices, faces, color, alpha=alpha)
        if collection:
            ax.add_collection3d(collection)
        
        # Add to legend
        edge_color = get_edge_color(color)
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=8, 
                                    label=f'{label} (vertices)'))
        legend_elements.append(Line2D([0], [0], color=edge_color, 
                                    linewidth=2, label=f'{label} (wireframe)'))
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Set axis limits based on all vertices
    if all_vertices:
        setup_3d_axis_equal_aspect(ax, all_vertices)
    
    # Configure labels and title
    title_text = f'Multiple Meshes Visualization ({len(meshes)} meshes)'
    configure_axis_labels(ax, show_labels=show_labels, 
                       title=title_text if show_labels else None,
                       fontsize_labels=12, fontsize_title=14)
    
    # Set figure title
    toggle_figure_title(fig, title_text, show_title, fontsize=16)
    
    # Configure grid
    configure_3d_grid(ax, show_grid=show_grid)
    
    # Add legend
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Show the plot
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure_if_path_provided(fig, save_path)
    
    plt.show()
    
    return fig, ax

