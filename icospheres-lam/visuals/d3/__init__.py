"""
3D visualization module for icospheres, meshes, and globe data visualization.

This module provides functions and classes for visualizing icospheres, mesh data, 
and geographic data in 3D space using matplotlib.

Functions
---------
visualize_icosphere_3d
    Visualize a single icosphere in 3D space
visualize_icosphere_layered_3d
    Visualize multiple icosphere refinement layers in a grid layout
visualize_multiple_meshes_3d
    Visualize multiple meshes in 3D space
globe_data_visualization
    Create global map visualization with interactive time slider

Classes
-------
GlobeGraphDataVisualizer
    Interactive 3D visualization class for mapping dataset values to lat-lon points on a globe
"""

from .visualize_icosphere_3d import visualize_icosphere_3d
from .visualize_icosphere_layered_3d import visualize_icosphere_layered_3d
from .visualize_multiple_meshes_3d import visualize_multiple_meshes_3d

__all__ = [
    'visualize_icosphere_3d',
    'visualize_icosphere_layered_3d',
    'visualize_multiple_meshes_3d',
]
