import matplotlib.pyplot as plt
from shapely import wkt
from ..utils import configure_axis_labels, crop_latlon_map, save_figure_if_path_provided, setup_cartopy_map, toggle_figure_title

def visualize_wkt(wkt_data, title="Country Shapes from WKT Data", show_title=True, show_labels=True, crop_bounds=None, save_path=None):
    """
    Visualize country polygons defined in WKT format on a 2D world map.
    
    Parameters
    ----------
    wkt_data : dict or list
        If dict: {country_name: wkt_string, ...}
        If list: [wkt_string, ...]
    title : str, optional
        Title for the plot, by default "Country Shapes from WKT Data"
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
    
    # Create figure and axis with projection
    fig, ax, projection = setup_cartopy_map(None, figsize=(14, 10))
    
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
    
    # Apply cropping if specified, otherwise show the whole world
    if crop_bounds is not None:
        crop_latlon_map(ax, crop_bounds)
    else:
        # Set extent to show the whole world
        ax.set_global()
    
    # Set title
    fig = plt.gcf()
    toggle_figure_title(fig, title, show_title, fontsize=14, fontweight='bold')
    
    # Configure axis labels (note: for Cartopy axes, this mainly affects title display)
    configure_axis_labels(ax, show_labels=show_labels, title=None)  # Title handled by toggle_figure_title
    
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure_if_path_provided(fig, save_path)
    
    plt.show()
