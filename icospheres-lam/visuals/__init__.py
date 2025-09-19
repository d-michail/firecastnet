"""
Visualization package for icospheres, meshes, and geographic data.

This package provides comprehensive visualization capabilities for icospheres, mesh data,
and geographic polygons in both 2D (latitude-longitude projection) and 3D space.

Submodules
----------
d2 : module
    2D visualization functions for lat-lon projections with world map backgrounds
d3 : module  
    3D visualization functions and classes for interactive globe visualizations
utils : module
    Utility functions for coordinate transformations, plotting setup, and data processing

Examples
--------
>>> from visuals.d2 import visualize_icosphere_latlon_2d
>>> from visuals.d3 import visualize_icosphere_3d, GlobeGraphDataVisualizer
>>> from visuals import utils
"""

# Import utils - this should work
try:
    from . import utils
except ImportError:
    # Fallback for when run as script
    import utils

# Import the d2 and d3 modules - now they have valid Python identifiers
try:
    from . import d2, d3
except ImportError as e:
    # Fallback for when run as script
    print(f"Warning: Could not import d2 or d3 submodules: {e}")
    try:
        import d2, d3
    except ImportError as e:
        print(f"Warning: Could not import visualization submodules: {e}")
        d2 = None
        d3 = None

__all__ = [
    'utils',
    'd2', 
    'd3'
]
