import matplotlib
import matplotlib.axes
import numpy as np
import matplotlib.pyplot as plt

def order(order, thing):
    return f"order_{order}_{thing}"

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

def get_edge_color(color: str, default=(0,0,0,0.825)):
    """
    Get the edge color based on the face color.
    
    Parameters
    ----------
    color : str
        The face color as a string (e.g., 'cyan', 'red').
    default : tuple, optional
        Default edge color if no specific mapping is found, by default (0, 0, 0, 0.825).
        
    Returns
    -------
    tuple
        The corresponding edge color as an RGBA tuple.
    """
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
    
    return edge_color_map.get(color, default)

def get_colors(amount: int):
    """
    Get a list of randomly generated bright colors using HSV color space.
    
    Parameters
    ----------
    amount : int
        Number of colors to generate.
    
    Returns
    -------
    list
        A list of color strings in hexadecimal format.
    """
    import random
    import colorsys
    colors = []
    for _ in range(amount):
        # Random hue (0-1), high saturation and value for brightness
        hue = random.random()
        saturation = 0.8  # High saturation for vivid colors
        value = 0.9       # High value for brightness
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Convert to hex format
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    
    return colors

def extract_icosphere_orders(icosphere_layers):
    """
    Extract available orders from the icosphere layers dictionary.
    
    Parameters:
    -----------
    icosphere_layers : dict
        Dictionary containing icosphere data with keys like 'order_X_vertices'
        
    Returns:
    --------
    list
        Sorted list of available order numbers
    """
    available_orders = []
    for key in icosphere_layers.keys():
        if key.endswith('_vertices'):
            order_num = int(key.split('_')[1])
            available_orders.append(order_num)
    
    return sorted(set(available_orders))

def setup_cartopy_map(fig, figsize=(12, 8), subplot_args=(1, 1, 1), features=True):
    """
    Set up a Cartopy map with common features and projection.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure or None
        Figure to add subplot to. If None, creates a new figure.
    figsize : tuple
        Figure size (width, height) in inches
    subplot_args : tuple
        Arguments for add_subplot (nrows, ncols, index)
    features : bool
        Whether to add common map features (coastlines, borders, etc.)
        
    Returns:
    --------
    tuple
        (fig, ax, projection) - Figure, axis, and projection objects
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    if fig is None:
        fig = plt.figure(figsize=figsize)
    
    projection = ccrs.PlateCarree()
    ax = fig.add_subplot(*subplot_args, projection=projection)
    
    if features:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.7)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.3, alpha=0.5)
        ax.add_feature(cfeature.LAND, alpha=0.1, color='lightgray')
        ax.add_feature(cfeature.OCEAN, alpha=0.1, color='lightblue')
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    
    return fig, ax, projection

def setup_3d_axis_equal_aspect(ax, vertices_list):
    """
    Set up 3D axis with equal aspect ratio based on vertex data.
    
    Parameters:
    -----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        3D axis to configure
    vertices_list : list of numpy.ndarray or numpy.ndarray
        List of vertex arrays or single vertex array to determine bounds
    """
    # Combine all vertices to find overall bounds
    if isinstance(vertices_list, list):
        all_points = np.vstack(vertices_list)
    else:
        all_points = vertices_list
    
    # Calculate the maximum range across all dimensions
    max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                         all_points[:, 1].max() - all_points[:, 1].min(),
                         all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0

    # Calculate center points
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

    # Set equal aspect ratio
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def configure_axis_labels(ax, show_labels=True, title=None, fontsize_labels=8, 
                         fontsize_title=10, fontsize_ticks=6):
    """
    Configure axis labels, title, and tick labels for both 2D and 3D axes.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes or matplotlib.axes._subplots.Axes3DSubplot
        2D or 3D axis to configure
    show_labels : bool
        Whether to show axis labels and title
    title : str, optional
        Title text to display. If None and show_labels=True, no title is set.
    fontsize_labels : int
        Font size for axis labels
    fontsize_title : int
        Font size for title
    fontsize_ticks : int
        Font size for tick labels
    """
    # Check if this is a 3D axis
    is_3d = hasattr(ax, 'zaxis')
    
    if show_labels:
        ax.set_xlabel('X', fontsize=fontsize_labels)
        ax.set_ylabel('Y', fontsize=fontsize_labels)
        
        if is_3d:
            ax.set_zlabel('Z', fontsize=fontsize_labels)
        
        if title is not None:
            pad = 25 if is_3d else None
            ax.set_title(title, fontsize=fontsize_title, fontweight='bold', pad=pad)
        
        # Configure tick labels
        ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize_ticks)
        
        if is_3d:
            ax.tick_params(axis='z', which='major', labelsize=fontsize_ticks)
            ax.tick_params(axis='z', which='minor', labelsize=fontsize_ticks)
    else:
        # Hide axis labels and title
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        
        if is_3d:
            ax.set_zlabel('')
            # Hide tick labels for 3D
            ax.tick_params(axis='both', which='major', labelbottom=False, 
                          labelleft=False, labelright=False, labeltop=False)
        else:
            # Hide tick labels for 2D
            ax.tick_params(axis='both', which='major', labelbottom=False, 
                          labelleft=False)

def configure_3d_grid(ax: matplotlib.axes.Axes, show_grid=True, alpha=0.3):
    """
    Configure 3D axis grid display.
    
    Parameters:
    -----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        3D axis to configure
    show_grid : bool
        Whether to show the grid
    alpha : float
        Grid transparency
    """
    if show_grid:
        ax.grid(True, alpha=alpha)
    else:
        ax.grid(False)
        ax.set_axis_off()  # Hide the axis if grid is not shown

def toggle_figure_title(fig, title_text=None, show_title=True, fontsize=12, fontweight='bold', **kwargs):
    """
    Toggle the title display on a matplotlib figure.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to modify the title for
    title_text : str, optional
        Title text to display when show_title is True. If None, no title is set.
    show_title : bool
        Whether to show the title or hide it
    fontsize : int
        Font size for the title
    fontweight : str
        Font weight for the title ('normal', 'bold', etc.)
    **kwargs
        Additional keyword arguments passed to fig.suptitle()
        
    Returns:
    --------
    matplotlib.text.Text or None
        The title text object if title is shown, None otherwise
    """
    if show_title and title_text is not None:
        return fig.suptitle(title_text, fontsize=fontsize, fontweight=fontweight, **kwargs)
    else:
        # Remove/hide the title
        fig.suptitle('')
        return None

def configure_cartopy_grid(ax, show_labels=True, position='all', alpha=0.5, linewidth=0.3):
    """
    Configure Cartopy axis gridlines and labels.
    
    Parameters:
    -----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        Cartopy axis to configure
    show_labels : bool
        Whether to show grid labels
    position : str
        Where to show labels: 'all', 'left', 'bottom', 'none'
    alpha : float
        Grid line transparency
    linewidth : float
        Grid line width
        
    Returns:
    --------
    gl : cartopy.mpl.gridliner.Gridliner
        The gridlines object for further customization
    """
    gl = ax.gridlines(draw_labels=show_labels, linewidth=linewidth, alpha=alpha)
    
    if show_labels and position != 'all':
        if position == 'left':
            gl.right_labels = False
            gl.top_labels = False
        elif position == 'bottom':
            gl.left_labels = False
            gl.top_labels = False
        elif position == 'none':
            gl.left_labels = False
            gl.right_labels = False
            gl.top_labels = False
            gl.bottom_labels = False
    
    return gl



def create_3d_polygon_collection(vertices_3d, faces, face_colors):
    """
    Create a 3D mesh collection for matplotlib plotting.
    
    Parameters:
    -----------
    vertices_3d : numpy.ndarray
        3D vertices array
    faces : numpy.ndarray
        Face indices array
    face_colors : numpy.ndarray or list
        Colors for each face
        
    Returns:
    --------
    Poly3DCollection
        3D polygon collection ready for plotting
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # Create triangular faces for the collection
    triangles = vertices_3d[faces]
    
    # Create the 3D collection
    collection = Poly3DCollection(triangles, alpha=0.7)
    collection.set_facecolors(face_colors)
    collection.set_edgecolor('black')
    collection.set_linewidth(0.1)
    
    return collection

def create_3d_mesh_collection(vertices, faces, color, show_wireframe=True, alpha=0.7, edge_alpha=0.825):
    """
    Create a 3D triangle collection for mesh visualization.
    
    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (N, 3) with vertex coordinates.
    faces : np.ndarray
        Array of shape (M, 3) with face indices.
    color : str
        Face color for the mesh.
    show_wireframe : bool, optional
        Whether to show wireframe edges, by default True.
    alpha : float, optional
        Transparency of the mesh, by default 0.7.
        
    Returns
    -------
    Poly3DCollection or None
        The 3D collection object ready to be added to an axis, or None if no faces.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # Create triangular faces
    triangles = []
    for face in faces:
        triangle = [vertices[face[j]] for j in range(3)]
        triangles.append(triangle)
    
    # Create and return collection if triangles exist
    if triangles:
        edge_color = get_edge_color(color, (0,0,0, edge_alpha))

        collection = Poly3DCollection(
            triangles,
            facecolors=color,
            edgecolors=edge_color,
            linewidths=0.5 if show_wireframe else 0,
            alpha=alpha
        )
        return collection
    
    return None


def crop_latlon_map(ax, crop_bounds):
    """
    Crop a latitude-longitude 2D map based on specified bounds.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        The axis containing the map to crop.
    crop_bounds : dict
        Dictionary containing crop parameters with keys 'lat' and 'lon'.
        Each value can be either:
        - A list/array [min, max] specifying the range
        - A single number N to crop from [-N, N]
        
    Examples
    --------
    # Crop to specific lat/lon ranges
    crop_latlon_map(ax, {'lat': [20, 60], 'lon': [-10, 40]})
    
    # Crop to symmetric ranges around 0
    crop_latlon_map(ax, {'lat': 45, 'lon': 90})
    
    # Mix of specific range and symmetric
    crop_latlon_map(ax, {'lat': [-30, 30], 'lon': 180})
    
    Returns
    -------
    None
        Modifies the axis in place.
    """
    # Process latitude bounds
    if 'lat' in crop_bounds:
        lat_val = crop_bounds['lat']
        if isinstance(lat_val, (list, tuple, np.ndarray)):
            # Specific range [min, max]
            if len(lat_val) == 2:
                lat_min, lat_max = lat_val
            else:
                raise ValueError("Latitude array must have exactly 2 values [min, max]")
        else:
            # Symmetric range [-N, N]
            lat_min, lat_max = -abs(lat_val), abs(lat_val)
        
        # Clamp to valid latitude range
        lat_min = max(lat_min, -90)
        lat_max = min(lat_max, 90)
        
        ax.set_ylim(lat_min, lat_max)
    
    # Process longitude bounds
    if 'lon' in crop_bounds:
        lon_val = crop_bounds['lon']
        if isinstance(lon_val, (list, tuple, np.ndarray)):
            # Specific range [min, max]
            if len(lon_val) == 2:
                lon_min, lon_max = lon_val
            else:
                raise ValueError("Longitude array must have exactly 2 values [min, max]")
        else:
            # Symmetric range [-N, N]
            lon_min, lon_max = -abs(lon_val), abs(lon_val)
        
        # Clamp to valid longitude range
        lon_min = max(lon_min, -180)
        lon_max = min(lon_max, 180)
        
        ax.set_xlim(lon_min, lon_max)


def save_figure_if_path_provided(fig, save_path=None, dpi=300, bbox_inches='tight', 
                                 create_dirs=True, **kwargs):
    """
    Save a matplotlib figure to a file if a save path is provided.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    save_path : str, optional
        Path where to save the figure (relative or absolute). If None, no saving occurs.
        File extension determines the format (e.g., .png, .pdf, .svg, .jpg).
    dpi : int, optional
        Resolution in dots per inch, by default 300.
    bbox_inches : str, optional
        Bounding box setting for saving, by default 'tight'.
    create_dirs : bool, optional
        Whether to create parent directories if they don't exist, by default True.
    **kwargs
        Additional keyword arguments passed to fig.savefig().
        
    Returns
    -------
    str or None
        The path where the figure was saved, or None if no save_path was provided.
        
    Examples
    --------
    # Basic usage
    save_figure_if_path_provided(fig, 'output/my_plot.png')
    
    # With custom settings
    save_figure_if_path_provided(fig, 'figures/plot.pdf', dpi=150, facecolor='white')
    
    # No saving (returns None)
    save_figure_if_path_provided(fig, None)
    """
    if save_path is None:
        return None
    
    import os
    
    # Create directories if they don't exist and create_dirs is True
    if create_dirs:
        dir_path = os.path.dirname(save_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    
    # Save the figure
    try:
        fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        print(f"Figure saved to: {save_path}")
        return save_path
    except Exception as e:
        print(f"Error saving figure to {save_path}: {e}")
        return None
