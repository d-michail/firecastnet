"""
Icospheres-LAM: Icosphere generation with adaptive mesh refinement.

This package provides tools for generating icospheres with localized adaptive mesh 
refinement based on geographic regions and polygon structures.

Core Classes
------------
PolygonStructure
    Data structure for defining polygon-based refinement targets

Core Functions  
--------------
generate_icosphere_file_code
    Generate file codes for icosphere configurations
load_yaml
    Load YAML configuration files
order
    Generate order-based keys for icosphere data structures

Modules
-------
visuals
    Visualization tools for 2D and 3D icosphere rendering
utils
    Utility functions for coordinate transformations and data processing
preprocess
    Preprocessing functions for polygon structures and mesh generation
"""

from .PolygonStructure import PolygonStructure
from .utils import generate_icosphere_file_code, load_yaml, order
from . import visuals

__all__ = [
    'PolygonStructure',
    'generate_icosphere_file_code', 
    'load_yaml',
    'order',
    'visuals'
]
