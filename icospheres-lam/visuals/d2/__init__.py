"""
2D visualization module for icospheres and meshes in latitude-longitude projection.

This module provides functions for visualizing icospheres, mesh data, and geographic
polygons in 2D using latitude-longitude projections with world map backgrounds.

Functions
---------
visualize_icosphere_latlon_2d
    Visualize a single icosphere in 2D lat-lon projection
visualize_icosphere_layered_latlon_2d
    Visualize multiple icosphere refinement layers in a grid layout
visualize_multiple_meshes_latlon_2d
    Visualize multiple meshes in 2D lat-lon projection
visualize_wkt
    Visualize country polygons from WKT format data
"""

from .visualize_icosphere_latlon_2d import visualize_icosphere_latlon_2d
from .visualize_icosphere_layered_latlon_2d import visualize_icosphere_layered_latlon_2d
from .visualize_multiple_meshes_latlon_2d import visualize_multiple_meshes_latlon_2d
from .visualize_wkt import visualize_wkt

__all__ = [
    'visualize_icosphere_latlon_2d',
    'visualize_icosphere_layered_latlon_2d', 
    'visualize_multiple_meshes_latlon_2d',
    'visualize_wkt'
]
