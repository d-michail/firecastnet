import pymesh
import numpy as np
from typing import List
from shapely import convex_hull
from shapely.geometry import Polygon, MultiPolygon
from buffers import buffer_polygon
from utils import get_icosahedron_geometry, to_cartesian, to_lat_lon, to_sphere
from PolygonStructure import PolygonStructure

def apply_buffers(polygon: PolygonStructure, base_refinement_order: int = 0) -> List[PolygonStructure]:
    """
    Apply buffers to the polygon based on the refinement order and buffer factor.
    
    :param polygon: The polygon structure to apply buffers to.
    :param base_refinement_order: The base refinement order.
    
    :return: A list of new polygons with applied buffers.
    """
    
    buffer_factor = polygon.buffer_factor
    buffer_unit = polygon.buffer_unit
    polygon_ref_order = polygon.refinement_order

    new_polygons = []
    for ref_order in range(base_refinement_order + 1, polygon_ref_order):
        buffer_size = buffer_factor * (polygon_ref_order - ref_order)
        new_polygon = buffer_polygon(polygon.wkt, buffer_size, buffer_unit)
        
        new_poly_struct = polygon.copy()
        new_poly_struct.wkt = new_polygon
        new_poly_struct.refinement_order = ref_order

        new_polygons.append(new_poly_struct)
    
    return new_polygons

def flatten_polygon_structures(polygon_structures: List[PolygonStructure]) -> List[PolygonStructure]:    
    flattened = []
    for polygon_struct in polygon_structures:
        # In case of a global polygon, we don't need to flatten it
        if polygon_struct.wkt is None:
            flattened.append(polygon_struct)
            continue

        poly = polygon_struct.wkt

        if isinstance(poly, MultiPolygon):
            for sub_polygon in poly.geoms:
                new_poly_struct = polygon_struct.copy()
                new_poly_struct.wkt = sub_polygon
                flattened.append(new_poly_struct)
        elif isinstance(poly, Polygon):
            flattened.append(polygon_struct)
    return flattened

def polygon_structures_preprocess(polygon_structures: List['PolygonStructure'], base_refinement_order: int = 0) -> List['PolygonStructure']:
    # Do the necessary reshaping and/or buffering of the polygons
    new_polygons = []
    for polygon in polygon_structures:
        refinement_type = polygon.refinement_type
        if refinement_type == "uniform":
            new_polygons.extend(apply_buffers(polygon, base_refinement_order))
        elif refinement_type == "block":
            polygon.wkt = convex_hull(polygon.wkt)

    polygon_structures.extend(new_polygons)

    # Extract all polygons
    polygon_structures = flatten_polygon_structures(polygon_structures)
    
    # Sort the polygon structure by refinement order
    polygon_structures = sorted(polygon_structures, key=lambda x: x.refinement_order)

    print(polygon_structures)
    return polygon_structures

e = 1e-5
def generate_initial_mesh(refinement_order=0, radius=1.0, center=(0.0, 0.0, 0.0)):
    # Get the initial icosahedron vertices and faces
    vertices, faces = get_icosahedron_geometry()

    # Initial mesh generation and subdivision
    mesh = pymesh.form_mesh(vertices, faces)
    mesh = pymesh.subdivide(mesh, refinement_order)
    mesh = pymesh.form_mesh(to_sphere(mesh.vertices, radius=radius, center=center), mesh.faces)

    vertices = to_lat_lon(mesh.vertices)
    # Adjust longitudes that are exactly 180.0
    vertices[np.isclose(vertices[:, 1], 180.0), 1] -= e

    # # Adjust longitudes that are exactly -180.0
    vertices[np.isclose(vertices[:, 1], -180.0), 1] += e
    
    print("Generated initial mesh with refinement order:", refinement_order)
    return pymesh.form_mesh(to_cartesian(vertices), mesh.faces)