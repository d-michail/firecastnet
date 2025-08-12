import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from utils import order

def to_lat_lon(vertices):
    y = vertices[:, 1]
    x = vertices[:, 0]
    z = vertices[:, 2]
    hyp = np.hypot(x, y)
    lat_rad = np.arctan2(z, hyp)
    lon_rad = np.arctan2(y, x)
    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)
    latlon = np.column_stack((lat_deg, lon_deg))
    return latlon

def face_wraps_around_latlon(triangle_vertices_latlon):
    """
    Determine if an icosphere triangle face wraps around the longitude boundary (-180°/180°)
    
    Args:
        triangle_vertices_latlon (np.ndarray): Array of shape (3, 2) containing the
            latitude and longitude coordinates of the triangle vertices.
            
    Returns:
        bool: True if the triangle wraps around the longitude boundary, False otherwise.
    """
    # Extract longitudes of the three vertices
    lons = [lon for _, lon in triangle_vertices_latlon]    
    
    # Calculate the maximum difference between longitudes
    max_diff = max(lons) - min(lons)

    # If the difference exceeds 180 degrees, it wraps around the dateline
    return max_diff > 180

def visualize_multiple_meshes_3d(meshes, colors=None, alpha=0.7, figsize=(12, 10)):
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
        
    Returns
    -------
    fig, ax : matplotlib figure and axis objects
        The created figure and 3D axis for further customization
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.lines import Line2D
    import numpy as np
    
    # Default colors if none provided
    if colors is None:
        default_colors = ['cyan', 'red', 'green', 'yellow', 'purple', 'orange', 
                         'pink', 'brown', 'gray', 'olive']
        colors = [default_colors[i % len(default_colors)] for i in range(len(meshes))]
    
    # Ensure we have enough colors
    if len(colors) < len(meshes):
        colors.extend(['blue'] * (len(meshes) - len(colors)))
    
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
        
        # Create triangular faces
        triangles = []
        for face in faces:
            triangle = [vertices[face[j]] for j in range(3)]
            triangles.append(triangle)
        
        # Add faces with wireframe
        if triangles:
            # Determine edge color (darker version of face color)
            edge_color_map = {
                'cyan': 'darkblue',
                'red': 'darkred',
                'green': 'darkgreen',
                'yellow': 'orange',
                'purple': 'indigo',
                'orange': 'darkorange',
                'pink': 'deeppink',
                'brown': 'saddlebrown',
                'gray': 'black',
                'olive': 'darkolivegreen',
                'blue': 'navy'
            }
            edge_color = edge_color_map.get(color, 'black')
            
            collection = Poly3DCollection(triangles, 
                                        facecolors=color, 
                                        edgecolors=edge_color,
                                        linewidths=0.5,
                                        alpha=alpha)
            ax.add_collection3d(collection)
        
        # Add to legend
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=8, 
                                    label=f'{label} (vertices)'))
        legend_elements.append(Line2D([0], [0], color=edge_color, 
                                    linewidth=2, label=f'{label} (wireframe)'))
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Set axis limits based on all vertices
    if all_vertices:
        all_points = np.vstack(all_vertices)
        
        max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                             all_points[:, 1].max() - all_points[:, 1].min(),
                             all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0
        
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'Multiple Meshes Visualization ({len(meshes)} meshes)')
    
    # Add legend
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def visualize_multiple_meshes_latlon_2d(meshes, colors=None, labels=None, alpha=0.7, figsize=(15, 10),
                                       show_wireframe=True, title=None):
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
        
    Returns
    -------
    fig, ax : matplotlib figure and axis objects
        The created figure and cartopy axis for further customization
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    # Default colors if none provided
    if colors is None:
        default_colors = ['cyan', 'red', 'green', 'yellow', 'purple', 'orange', 
                         'pink', 'brown', 'gray', 'olive', 'blue', 'magenta']
        colors = [default_colors[i % len(default_colors)] for i in range(len(meshes))]
    
    # Ensure we have enough colors
    if len(colors) < len(meshes):
        colors.extend(['blue'] * (len(meshes) - len(colors)))
    
    # Create figure and axis with cartopy projection
    fig = plt.figure(figsize=figsize)
    projection = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.7)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.3, alpha=0.5)
    ax.add_feature(cfeature.LAND, alpha=0.1, color='lightgray')
    ax.add_feature(cfeature.OCEAN, alpha=0.1, color='lightblue')
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    
    legend_elements = []
    
    # Process each mesh
    for i, mesh in enumerate(meshes):
        vertices_3d = mesh['vertices']
        faces = mesh['faces']

        label = labels[i] if labels else f'Mesh {i+1}'
        color = colors[i]
        
        # Convert 3D vertices to lat-lon
        vertices_latlon = to_lat_lon(vertices_3d)
        
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
            edge_color_map = {
                'cyan': 'darkblue',
                'red': 'darkred',
                'green': 'darkgreen',
                'yellow': 'orange',
                'purple': 'indigo',
                'orange': 'darkorange',
                'pink': 'deeppink',
                'brown': 'saddlebrown',
                'gray': 'black',
                'olive': 'darkolivegreen',
                'blue': 'navy',
                'royalblue': 'darkblue',
                'lightblue': 'blue',
                'magenta': 'darkmagenta'
            }
            edge_color = edge_color_map.get(color, (0,0,0,0.7))
            
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
    
    # Set extent to show the whole world
    ax.set_global()
    
    # Add title
    if title is None:
        title = f'Multiple Meshes Visualization - World Map ({len(meshes)} meshes)'
    plt.title(title, fontweight='bold', pad=20, fontsize=14)

    # Add legend
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def visualize_icosphere_3d(icosphere, refinement_order=0, show_wireframe=True, alpha=0.7, 
                                  polygon_vertices=None, polygon_centroid=None,
                                  highlighted_faces=None, highlight_color='red',
                                  show_face_numbers=False):
    """
    Visualize an icosphere using Matplotlib, optionally with a spherical polygon and its centroid,
    and the ability to highlight specific faces.

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
    highlighted_faces : list, optional
        List of face indices to highlight with a different color
    highlight_color : str, optional
        Color to use for highlighted faces, by default 'red'
    show_face_numbers : bool, optional
        Whether to display face indices in the triangles, by default False

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

    # Create triangular faces
    regular_triangles = []
    highlighted_triangles = []
    face_centers = []
    face_indices = []

    # Create a set of highlighted faces for faster lookup
    highlighted_faces_set = set() if highlighted_faces is None else set(highlighted_faces)

    for i, face in enumerate(faces):
        triangle = [vertices[face[j]] for j in range(3)]
        if i in highlighted_faces_set:
            highlighted_triangles.append(triangle)
        else:
            regular_triangles.append(triangle)

        # Calculate and store the face center and index if we're showing face numbers
        if show_face_numbers:
            # Calculate the center of the triangle
            center = np.mean(triangle, axis=0)
            # Normalize to push slightly outside the sphere surface
            center_normalized = center / np.linalg.norm(center) * 1.01
            face_centers.append(center_normalized)
            face_indices.append(i)

    # Add regular triangles to the plot
    if regular_triangles:
        collection = Poly3DCollection(regular_triangles, 
                                     facecolors='cyan', 
                                     edgecolors='darkblue' if show_wireframe else None,
                                     linewidths=0.5 if show_wireframe else 0,
                                     alpha=alpha)
        collection.set_facecolor((0, 0.5, 0.7, alpha))
        ax.add_collection3d(collection)

    # Add highlighted triangles to the plot
    if highlighted_triangles:
        highlight_collection = Poly3DCollection(highlighted_triangles, 
                                              facecolors=highlight_color, 
                                              edgecolors='darkred' if show_wireframe else None,
                                              linewidths=0.5 if show_wireframe else 0,
                                              alpha=alpha)
        ax.add_collection3d(highlight_collection)

    # Display face numbers if requested
    if show_face_numbers:
        for center, face_idx in zip(face_centers, face_indices):
            ax.text(center[0], center[1], center[2], 
                   str(face_idx), color='black', fontsize=8, 
                   ha='center', va='center', fontweight='bold')

    # Initialize legend elements
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=8, label='Icosphere Vertices'),
        Patch(facecolor='cyan', edgecolor='darkblue', alpha=alpha, label='Icosphere Faces')
    ]

    # Add highlighted faces to legend if present
    if highlighted_triangles:
        legend_elements.append(Patch(facecolor=highlight_color, edgecolor='darkred', 
                                    alpha=alpha, label='Highlighted Faces'))

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

    max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                         all_points[:, 1].max() - all_points[:, 1].min(),
                         all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0

    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Add title and labels
    title = f'Icosphere (Order {refinement_order})'
    if polygon_vertices is not None:
        title += ' with Spherical Polygon'
    if highlighted_faces:
        title += f' and {len(highlighted_faces)} Highlighted Faces'
    if show_face_numbers:
        title += ' (with face indices)'
    plt.title(title)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add legend
    ax.legend(handles=legend_elements, loc='upper right')

    # Show the plot
    plt.tight_layout()
    plt.show()

def visualize_icosphere_latlon_2d(icospheres, refinement_order=0, show_wireframe=True, alpha=0.7, 
                                polygon_vertices=None, polygon_centroid=None,
                                highlighted_faces=None, highlight_color='red',
                                show_face_numbers=False):
    """
    Visualize an icosphere in 2D latitude-longitude space with a world map background.
    
    Parameters
    ----------
    icospheres : dict
        Dictionary containing icosphere data
    order : int, optional
        The order of the icosphere to visualize, by default 0
    show_wireframe : bool, optional
        Whether to show the wireframe of the icosphere, by default True
    alpha : float, optional
        Transparency of the surface, by default 0.7
    polygon_vertices : np.ndarray, optional
        Array of shape (N, 3) representing the polygon vertices in 3D
    polygon_centroid : np.ndarray, optional
        Array of shape (3,) representing the polygon centroid in 3D
    highlighted_faces : list, optional
        List of face indices to highlight with a different color
    highlight_color : str, optional
        Color to use for highlighted faces, by default 'red'
    show_face_numbers : bool, optional
        Whether to display face indices in the triangles, by default False
        
    Returns
    -------
    None
        Displays a matplotlib 2D visualization with world map background
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    
    # Get vertices and faces for the specified order
    vertices_3d = icospheres[order(refinement_order, "vertices")]
    faces = icospheres[order(refinement_order, "faces")]

    # Convert 3D vertices to lat-lon
    vertices_latlon = to_lat_lon(vertices_3d)
    
    # Create figure and axis with projection
    fig = plt.figure(figsize=(12, 8))
    
    # Use PlateCarree projection for simplicity
    projection = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, alpha=0.1)
    ax.add_feature(cfeature.OCEAN, alpha=0.1)
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    
    # Plot vertices
    ax.scatter(vertices_latlon[:, 1], vertices_latlon[:, 0], 
              color='b', s=10, alpha=0.6)
    
    # Create a set of highlighted faces for faster lookup
    highlighted_faces_set = set() if highlighted_faces is None else set(highlighted_faces)
    
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
        
        # Determine the color and edge color based on whether it's highlighted
        if i in highlighted_faces_set:
            facecolor = highlight_color
            edgecolor = 'darkred' if show_wireframe else None
        else:
            facecolor = 'cyan'
            edgecolor = 'darkblue' if show_wireframe else None
            
        # Plot the triangle
        ax.fill(triangle_lons, triangle_lats,
                facecolor=facecolor, 
                edgecolor=edgecolor,
                linewidth=0.5 if show_wireframe else 0,
                alpha=alpha)
        
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
        Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=8, label='Icosphere Vertices'),
        Patch(facecolor='cyan', edgecolor='darkblue' if show_wireframe else None, alpha=alpha, label='Icosphere Faces')
    ]
    
    # Add highlighted faces to legend if present
    if highlighted_faces:
        legend_elements.append(Patch(facecolor=highlight_color, edgecolor='darkred' if show_wireframe else None, 
                                    alpha=alpha, label='Highlighted Faces'))
    
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
    title = f'Icosphere (Order {refinement_order}) in Lat-Lon Projection'
    if polygon_vertices is not None:
        title += ' with Spherical Polygon'
    if highlighted_faces:
        title += f' and {len(highlighted_faces)} Highlighted Faces'
    if show_face_numbers:
        title += ' (with face indices)'
    plt.title(title)
    
    # Set extent to show the whole world
    ax.set_global()
    
    # Add legend
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def visualize_icosphere_layered_latlon_2d(icosphere_layers, max_columns=3, show_wireframe=True, alpha=0.7,
                                         figsize=(20, 12), colors=None, 
                                         title="Icosphere Refinement Layers - Lat-Lon Projection"):
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
    title : str, optional
        Main title for the entire figure, by default "Icosphere Refinement Layers - Lat-Lon Projection"
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object for further customization
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    # Extract available orders from the dictionary
    available_orders = []
    for key in icosphere_layers.keys():
        if key.endswith('_vertices'):
            order_num = int(key.split('_')[1])
            available_orders.append(order_num)
    
    available_orders = sorted(set(available_orders))
    num_layers = len(available_orders)
    
    if num_layers == 0:
        print("No icosphere layers found in the provided data.")
        return None
    
    # Calculate grid dimensions
    num_cols = min(max_columns, num_layers)
    num_rows = (num_layers + num_cols - 1) // num_cols  # Ceiling division
    
    # Default colors if none provided
    if colors is None:
        default_colors = ['cyan', 'red', 'green', 'yellow', 'purple', 'orange', 
                         'pink', 'brown', 'gray', 'olive', 'blue', 'magenta']
        colors = [default_colors[i % len(default_colors)] for i in range(num_layers)]
    
    # Ensure we have enough colors
    if len(colors) < num_layers:
        colors.extend(['blue'] * (num_layers - len(colors)))
    
    # Create figure with subplots using cartopy projection
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Use PlateCarree projection for all subplots
    projection = ccrs.PlateCarree()
    
    # Create subplots for each layer
    for i, o in enumerate(available_orders):
        # Create 2D subplot with cartopy projection
        ax = fig.add_subplot(num_rows, num_cols, i + 1, projection=projection)
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.7)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.3, alpha=0.5)
        ax.add_feature(cfeature.LAND, alpha=0.1, color='lightgray')
        ax.add_feature(cfeature.OCEAN, alpha=0.1, color='lightblue')
        ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)
        
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
            edge_color_map = {
                'cyan': 'darkblue',
                'red': 'darkred',
                'green': 'darkgreen',
                'yellow': 'orange',
                'purple': 'indigo',
                'orange': 'darkorange',
                'pink': 'deeppink',
                'brown': 'saddlebrown',
                'gray': 'black',
                'olive': 'darkolivegreen',
                'blue': 'navy',
                'magenta': 'darkmagenta'
            }
            edge_color = edge_color_map.get(color, 'black')
            
            # Plot the triangle
            ax.fill(triangle_lons, triangle_lats,
                    facecolor=color, 
                    edgecolor=edge_color if show_wireframe else None,
                    linewidth=0.1 if show_wireframe else 0,
                    alpha=alpha,
                    transform=projection)
        
        # Set extent to show the whole world
        ax.set_global()
        
        # Add subtitle for this subplot
        ax.set_title(f'Order {o}', 
                    fontsize=10, fontweight='bold', pad=15)
        
        # Add gridlines with labels only for edge subplots to avoid clutter
        if i % num_cols == 0:  # Left edge
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
            gl.right_labels = False
            gl.top_labels = False
        elif i >= num_layers - num_cols:  # Bottom edge
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
            gl.left_labels = False
            gl.top_labels = False
        else:
            ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)
    
    # Add a legend for the colors if there's space
    if num_layers < num_rows * num_cols:
        # If there's space for a legend subplot
        legend_ax = fig.add_subplot(num_rows, num_cols, num_layers + 1)
        legend_ax.axis('off')
        
        # Create legend elements
        legend_elements = []
        for i, o in enumerate(available_orders):
            color = colors[i]
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color, markersize=8, 
                      label=f'Order {o}')
            )
        
        legend_ax.legend(handles=legend_elements, loc='center', 
                        title='Refinement Orders', title_fontsize=12)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.1)  # More room for main title and spacing
    plt.show()
    
    return fig

def visualize_wkt(wkt_data, title="Country Shapes from WKT Data"):
    """
    Visualize country polygons defined in WKT format on a 2D world map.
    
    Parameters
    ----------
    wkt_data : dict or list
        If dict: {country_name: wkt_string, ...}
        If list: [wkt_string, ...]
    title : str, optional
        Title for the plot, by default "Country Shapes from WKT Data"
    
    Returns
    -------
    None
        Displays a matplotlib 2D visualization with world map background
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from shapely import wkt
    
    # Create figure and axis with projection
    fig = plt.figure(figsize=(14, 10))
    projection = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    
    # Add basic map features
    ax.add_feature(cfeature.COASTLINE, edgecolor='gray', alpha=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.1)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.1)
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    
    # Set up colors for multiple countries
    colors = plt.cm.tab20.colors
    
    # Process and plot the WKT data
    if isinstance(wkt_data, dict):
        items = wkt_data.items()
    else:
        items = enumerate(wkt_data)
    
    for i, (key, wkt_string) in enumerate(items):
        try:
            # Parse WKT string to geometry
            geometry = wkt.loads(wkt_string)
            
            # Set color based on index
            color = colors[i % len(colors)]
            
            # Handle different geometry types
            if geometry.geom_type == 'Polygon':
                y, x = geometry.exterior.xy
                ax.fill(x, y, color=color, alpha=0.6, transform=projection)
                ax.plot(x, y, color='black', linewidth=0.5, transform=projection)
                # Add label at centroid
                centroid = geometry.centroid
                if isinstance(key, str):
                    ax.text(centroid.x, centroid.y, key, 
                           fontsize=8, ha='center', transform=projection)
                
            elif geometry.geom_type == 'MultiPolygon':
                for polygon in geometry.geoms:
                    y, x = polygon.exterior.xy
                    ax.fill(x, y, color=color, alpha=0.6, transform=projection)
                    ax.plot(x, y, color='black', linewidth=0.5, transform=projection)
                
                # Add label at largest polygon centroid if it's a named country
                if isinstance(key, str):
                    # Find the largest polygon for label placement
                    largest_area = 0
                    largest_centroid = None
                    for polygon in geometry.geoms:
                        if polygon.area > largest_area:
                            largest_area = polygon.area
                            largest_centroid = polygon.centroid
                    
                    if largest_centroid:
                        ax.text(largest_centroid.x, largest_centroid.y, key, 
                               fontsize=8, ha='center', transform=projection)        
        except Exception as e:
            print(f"Error with {key}: {e}")
    
    # Set extent to show the whole world
    ax.set_global()
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

def visualize_icosphere_layered_3d(icosphere_layers, max_columns=3, show_wireframe=True, alpha=0.7,
                                  rotation=(30, 45), figsize=(15, 10), colors=None,
                                  zoom=None, title="Icosphere Refinement Layers"):
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
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object for further customization
    """
    # Extract available orders from the dictionary
    available_orders = []
    for key in icosphere_layers.keys():
        if key.endswith('_vertices'):
            order_num = int(key.split('_')[1])
            available_orders.append(order_num)
    
    available_orders = sorted(set(available_orders))
    num_layers = len(available_orders)
    
    if num_layers == 0:
        print("No icosphere layers found in the provided data.")
        return None
    
    # Calculate grid dimensions
    num_cols = min(max_columns, num_layers)
    num_rows = (num_layers + num_cols - 1) // num_cols  # Ceiling division
    
    # Default colors if none provided
    if colors is None:
        default_colors = ['cyan', 'red', 'green', 'yellow', 'purple', 'orange', 
                         'pink', 'brown', 'gray', 'olive', 'blue', 'magenta']
        colors = [default_colors[i % len(default_colors)] for i in range(num_layers)]
    
    # Ensure we have enough colors
    if len(colors) < num_layers:
        colors.extend(['blue'] * (num_layers - len(colors)))
    
    # Handle zoom factors
    if zoom is None:
        zoom = [1.0] * num_layers
    elif len(zoom) < num_layers:
        # Extend zoom list by repeating the last value
        last_zoom = zoom[-1] if zoom else 1.0
        zoom.extend([last_zoom] * (num_layers - len(zoom)))
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
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
        
        # Create triangular faces
        triangles = []
        for face in faces:
            triangle = [vertices[face[j]] for j in range(3)]
            triangles.append(triangle)
        
        # Add faces with wireframe
        if triangles:
            # Determine edge color (darker version of face color)
            edge_color_map = {
                'cyan': 'darkblue',
                'red': 'darkred',
                'green': 'darkgreen',
                'yellow': 'orange',
                'purple': 'indigo',
                'orange': 'darkorange',
                'pink': 'deeppink',
                'brown': 'saddlebrown',
                'gray': 'black',
                'olive': 'darkolivegreen',
                'blue': 'navy',
                'magenta': 'darkmagenta',
                'royalblue': 'darkblue'
            }
            edge_color = edge_color_map.get(color, 'black')
            
            if show_wireframe:
                collection = Poly3DCollection(triangles, 
                                            facecolors=color, 
                                            edgecolors=edge_color,
                                            linewidths=0.5,
                                            alpha=alpha)
            else:
                collection = Poly3DCollection(triangles, 
                                            facecolors=color, 
                                            alpha=alpha)
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
        
        # Add labels and title for this subplot
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        
        # Count vertices and faces for the title
        ax.set_title(f'Order {o}', 
                    fontsize=10, fontweight='bold', pad=25)  # Add padding between title and plot
        
        # Reduce tick label size to save space
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.tick_params(axis='both', which='minor', labelsize=6)
    
    # Add a legend for the colors in the remaining space or as a separate element
    # if num_layers < num_rows * num_cols:
    #     # If there's space for a legend subplot
    #     legend_ax = fig.add_subplot(num_rows, num_cols, num_layers + 1)
    #     legend_ax.axis('off')
        
        # # Create legend elements
        # legend_elements = []
        # for i, o in enumerate(available_orders):
        #     color = colors[i]
        #     legend_elements.append(
        #         Line2D([0], [0], marker='o', color='w', 
        #               markerfacecolor=color, markersize=6, 
        #               label=f'Order  {o}')
        #     )
        
        # legend_ax.legend(handles=legend_elements, loc='center', 
        #                 title='Refinement Orders', title_fontsize=10)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, hspace=0.3)  # More room for main title and spacing between rows
    plt.show()
    
    return fig

if __name__ == "__main__":    
    # Example usage for visualize_icosphere_layered_3d
    # Uncomment the following lines to test the layered visualization
    
    # import json
    # with open('path_to_your_icosphere_layers.json', 'r') as f:
    #     icosphere_layers = json.load(f)
    # 
    # # Example with different zoom levels for each subfigure (3D)
    # visualize_icosphere_layered_3d(
    #     icosphere_layers, 
    #     max_columns=2, 
    #     rotation=(20, 60),
    #     zoom=[0.8, 1.0, 1.2, 1.5],  # Progressive zoom: zoom out, normal, zoom in, zoom in more
    #     title="My Icosphere Refinement Progression with Individual Zoom"
    # )
    #
    # # Example for 2D lat-lon layered visualization
    # visualize_icosphere_layered_latlon_2d(
    #     icosphere_layers,
    #     max_columns=2,
    #     show_wireframe=True,
    #     title="Icosphere Refinement Layers - World Map View"
    # )
    
    # Original WKT visualization example
    import pandas as pd
    df = pd.read_csv('src/structured_icospheres/csv/extracted_countries.csv', encoding='latin1', quotechar='"')        
    visualize_wkt(
        df['WKT'].to_list(),
        title="Countries from WKT Data"
    )