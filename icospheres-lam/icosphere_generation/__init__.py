"""
Icosphere generation module containing core functionality for generating adaptive icospheres.

This module contains the following components:
- PolygonStructure: Type definition for polygon structures used in icosphere generation
- utils: Utility functions for coordinate transformations, icosphere generation, and mesh operations
- preprocess: Functions for preprocessing polygons and generating initial mesh structures
- buffers: Functions for applying geographic buffers to polygons
- validate_mesh: Functions for validating and debugging generated meshes
- mesh_operations: Functions for mesh manipulation, face intersection detection, and mesh stitching
- process: Functions for configuration processing and generation orchestration
"""

from .PolygonStructure import PolygonStructure
from .utils import (
    load_yaml, order, generate_icosphere_file_code, 
    undo_antimeridian_wrap, polygon_crosses_antimeridian,
    to_lat_lon, to_sphere, to_cartesian, mesh_to_dict,
    get_icosahedron_geometry, flatten_polygons
)
from .preprocess import generate_initial_mesh, polygon_structures_preprocess, apply_buffers
from .buffers import buffer_polygon, flip_transform
from .validate_mesh import _save_comprehensive_mesh_debug
from .mesh_operations import (
    find_intersecting_icosphere_faces, construct_submesh, 
    mesh_stitch, generate_icosphere
)
from .process import initialize_PolygonStructures, execute_icosphere_generation

__all__ = [
    'PolygonStructure',
    'load_yaml', 'order', 'generate_icosphere_file_code',
    'undo_antimeridian_wrap', 'polygon_crosses_antimeridian',
    'to_lat_lon', 'to_sphere', 'to_cartesian', 'mesh_to_dict',
    'get_icosahedron_geometry', 'flatten_polygons',
    'generate_initial_mesh', 'polygon_structures_preprocess', 'apply_buffers',
    'buffer_polygon', 'flip_transform',
    '_save_comprehensive_mesh_debug',
    'find_intersecting_icosphere_faces', 'construct_submesh', 
    'mesh_stitch', 'generate_icosphere',
    'initialize_PolygonStructures', 'execute_icosphere_generation'
]