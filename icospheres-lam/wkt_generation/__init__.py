"""
WKT Generation Package

This package contains utilities for generating Well-Known Text (WKT) geometries
from various geographic data sources including countries, continents, GFED regions,
and custom shapefiles.

Modules:
    geometry_utils: Utilities for coordinate transformation and geometry manipulation
    clustering: Point clustering algorithms for geographic data
    data_utils: Data loading and processing utilities
    countries: Country and continent geometry processing
    gfed: GFED region geometry processing
    shapefiles: Custom shapefile processing from local files or URLs
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main functions for easy access
from .geometry_utils import flip_coords, reshape_wkt_coords
from .clustering import points_clustering
from .data_utils import get_country
from .shapefiles import process_shapefiles

__all__ = [
    'flip_coords',
    'reshape_wkt_coords', 
    'points_clustering',
    'get_country',
    'process_shapefiles'
]