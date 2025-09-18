"""
Geometry Utilities for WKT Generation

This module contains utilities for coordinate transformation and geometry manipulation
used in the WKT generation process.
"""

import shapely
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform as shapely_transform


def flip_coords(x, y):
    """
    Flip coordinates from (x, y) to (y, x) format.
    
    Args:
        x: X coordinate (typically longitude)
        y: Y coordinate (typically latitude)
        
    Returns:
        tuple: Flipped coordinates (y, x)
    """
    return y, x


def reshape_wkt_coords(wkt_shape):
    """
    Flip the coordinates of a WKT shape to (lat, lon) format.
    
    This function transforms coordinates from (lon, lat) to (lat, lon) format
    for both Polygon and MultiPolygon geometries.
    
    Args:
        wkt_shape (Polygon or MultiPolygon): The geometry to transform
        
    Returns:
        Polygon or MultiPolygon: The transformed geometry with flipped coordinates
        
    Raises:
        ValueError: If the shape is neither a Polygon nor MultiPolygon
    """
    if isinstance(wkt_shape, Polygon):
        return shapely_transform(flip_coords, Polygon(wkt_shape))
    elif isinstance(wkt_shape, MultiPolygon):
        shapes = []
        for poly in wkt_shape.geoms:
            shapes.append(shapely_transform(flip_coords, Polygon(poly)))
        return MultiPolygon(shapes)
    else:
        raise ValueError("Shape must be a Polygon or MultiPolygon")