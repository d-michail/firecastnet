import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.collection import GeometryCollection
from typing import Union, List, Tuple, Optional
import random

def plot_shapes(
    geometries: Union[BaseGeometry, List[BaseGeometry]], 
    ax=None, 
    figsize: Tuple[int, int] = (10, 10),
    colors: Optional[List[str]] = None,
    alpha: float = 0.6,
    edgecolor: str = 'black',
    linewidth: float = 1.0,
    show_vertices: bool = False,
    vertex_size: int = 20,
    vertex_color: str = 'red',
    show_legend: bool = True,
    title: str = 'Shapely Geometries',
    display: bool = True,
    lonlat: bool = False
) -> plt.Axes:
    """
    Plot Shapely geometries with different colors for multiple shapes.
    
    Parameters:
    -----------
    geometries : BaseGeometry or List[BaseGeometry]
        Single shapely geometry or list of geometries to plot
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, new figure and axes are created
    figsize : Tuple[int, int], optional
        Figure size if creating a new figure
    colors : List[str], optional
        List of colors to use for geometries. If None, random colors are generated
    alpha : float, optional
        Alpha transparency for fill
    edgecolor : str, optional
        Color for shape edges
    linewidth : float, optional
        Width of shape edges
    show_vertices : bool, optional
        Whether to show vertices of polygons and lines
    vertex_size : int, optional
        Size of vertices if shown
    vertex_color : str, optional
        Color of vertices if shown
    show_legend : bool, optional
        Whether to show a legend for multiple shapes
    title : str, optional
        Title for the plot
    display : bool, optional
        Whether to display the plot immediately using plt.show()
    lonlat : bool, optional
        If True, coordinates are interpreted as (longitude, latitude).
        If False (default), coordinates are interpreted as (latitude, longitude).
    
    Returns:
    --------
    matplotlib.axes.Axes
        The axes object containing the plot
    """
    # Ensure geometries is a list
    if isinstance(geometries, (BaseGeometry, GeometryCollection)):
        geometries = [geometries]
    
    # Create new axes if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_new_fig = True
    else:
        created_new_fig = False
    
    # Generate colors if not provided
    if colors is None:
        colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(len(geometries))]
    elif len(colors) < len(geometries):
        # Extend colors if not enough provided
        additional_colors = [f"#{random.randint(0, 0xFFFFFF):06x}" 
                            for _ in range(len(geometries) - len(colors))]
        colors.extend(additional_colors)
    
    # Track bounds to set appropriate limits
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    
    # Plot each geometry
    for i, geom in enumerate(geometries):
        color = colors[i]
        label = f"Shape {i+1}"
        
        if geom.is_empty:
            continue
            
        # Handle different geometry types
        if isinstance(geom, Point):
            if lonlat:
                ax.scatter(geom.x, geom.y, color=color, s=100, alpha=alpha, 
                          edgecolor=edgecolor, linewidth=linewidth, label=label)
                min_x = min(min_x, geom.x)
                max_x = max(max_x, geom.x)
                min_y = min(min_y, geom.y)
                max_y = max(max_y, geom.y)
            else:
                # For latlon, we swap x and y for plotting
                ax.scatter(geom.y, geom.x, color=color, s=100, alpha=alpha, 
                          edgecolor=edgecolor, linewidth=linewidth, label=label)
                min_x = min(min_x, geom.y)
                max_x = max(max_x, geom.y)
                min_y = min(min_y, geom.x)
                max_y = max(max_y, geom.x)
            
        elif isinstance(geom, LineString):
            x, y = geom.xy
            if lonlat:
                ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, label=label)
                if show_vertices:
                    ax.scatter(x, y, color=vertex_color, s=vertex_size, zorder=10)
                min_x = min(min_x, min(x))
                max_x = max(max_x, max(x))
                min_y = min(min_y, min(y))
                max_y = max(max_y, max(y))
            else:
                # For latlon, we swap x and y for plotting
                ax.plot(y, x, color=color, linewidth=linewidth, alpha=alpha, label=label)
                if show_vertices:
                    ax.scatter(y, x, color=vertex_color, s=vertex_size, zorder=10)
                min_x = min(min_x, min(y))
                max_x = max(max_x, max(y))
                min_y = min(min_y, min(x))
                max_y = max(max_y, max(x))
            
        elif isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            if lonlat:
                ax.fill(x, y, alpha=alpha, color=color, edgecolor=edgecolor, 
                       linewidth=linewidth, label=label)
                
                if show_vertices:
                    ax.scatter(x, y, color=vertex_color, s=vertex_size, zorder=10)
                
                # Plot holes if any
                for hole in geom.interiors:
                    hx, hy = hole.xy
                    ax.fill(hx, hy, alpha=0, edgecolor=edgecolor, linewidth=linewidth)
                    
                    if show_vertices:
                        ax.scatter(hx, hy, color=vertex_color, s=vertex_size, zorder=10)
                
                # Update bounds
                min_x = min(min_x, geom.bounds[0])
                min_y = min(min_y, geom.bounds[1])
                max_x = max(max_x, geom.bounds[2])
                max_y = max(max_y, geom.bounds[3])
            else:
                # For latlon, we swap x and y for plotting
                ax.fill(y, x, alpha=alpha, color=color, edgecolor=edgecolor, 
                       linewidth=linewidth, label=label)
                
                if show_vertices:
                    ax.scatter(y, x, color=vertex_color, s=vertex_size, zorder=10)
                
                # Plot holes if any
                for hole in geom.interiors:
                    hx, hy = hole.xy
                    ax.fill(hy, hx, alpha=0, edgecolor=edgecolor, linewidth=linewidth)
                    
                    if show_vertices:
                        ax.scatter(hy, hx, color=vertex_color, s=vertex_size, zorder=10)
                
                # Update bounds (swap indices for latlon)
                min_x = min(min_x, geom.bounds[1])
                min_y = min(min_y, geom.bounds[0])
                max_x = max(max_x, geom.bounds[3])
                max_y = max(max_y, geom.bounds[2])
            
        else:  # For GeometryCollection or other types
            # Recursively plot each part with the same color
            for part in geom.geoms if hasattr(geom, 'geoms') else [geom]:
                plot_shapes(part, ax=ax, colors=[color], alpha=alpha, 
                           edgecolor=edgecolor, linewidth=linewidth,
                           show_vertices=show_vertices, vertex_size=vertex_size,
                           vertex_color=vertex_color, show_legend=False,
                           display=False, lonlat=lonlat)
            
            # Add single legend entry for the collection
            ax.plot([], [], color=color, label=label)
            
            # Update bounds from geom.bounds
            if hasattr(geom, 'bounds') and geom.bounds:
                if lonlat:
                    min_x = min(min_x, geom.bounds[0])
                    min_y = min(min_y, geom.bounds[1])
                    max_x = max(max_x, geom.bounds[2])
                    max_y = max(max_y, geom.bounds[3])
                else:
                    # Swap indices for latlon
                    min_x = min(min_x, geom.bounds[1])
                    min_y = min(min_y, geom.bounds[0])
                    max_x = max(max_x, geom.bounds[3])
                    max_y = max(max_y, geom.bounds[2])
    
    # Add padding to bounds
    if min_x != float('inf') and max_x != float('-inf'):
        padding = max(max_x - min_x, max_y - min_y) * 0.05
        ax.set_xlim([min_x - padding, max_x + padding])
        ax.set_ylim([min_y - padding, max_y + padding])
    
    # Set equal aspect ratio to avoid distortion
    ax.set_aspect('equal')
    
    # Add title and legend
    ax.set_title(title)
    if show_legend and len(geometries) > 1:
        ax.legend()
    
    # Display the plot if requested and if we created a new figure
    if display and created_new_fig:
        plt.tight_layout()
        plt.show()
    
    return ax

if __name__ == "__main__":
    # Example usage of the plot_shapes function

    # Create some example shapes
    point = Point(0, 0)
    triangle = Polygon([(0, 0), (1, 0), (1, 1)])
    square = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])

    # Plot them together
    plot_shapes([point, triangle, square])

