import copy
import pymesh
import numpy as np
from shapely import convex_hull
from shapely.geometry import Polygon, MultiPolygon
from shapely.wkt import loads as wkt_loads
from buffers import buffer_polygon_by_percent, buffer_polygon_in_km
from utils import get_icosahedron_geometry, to_cartesian, to_lat_lon, to_sphere

def apply_buffers(polygon, base_refinement_order=0):
    """
    Apply buffers to the polygon based on the refinement order and buffer factor.
    
    :param polygon: The polygon structure to apply buffers to.
    
    :return: A list of new polygons with applied buffers.
    """
    buffer_factor = polygon["buffer_factor"]
    buffer_unit = polygon["buffer_unit"]
    polygon_ro = polygon["refinement_order"]
    print(f"Processing polygon for polygon {polygon['target_code']}")
    new_polygons = []
    for ro in range(base_refinement_order + 1, polygon_ro):
        polygon_copy = copy.copy(polygon)
        buffer_size = buffer_factor * (polygon_ro - ro)
        if buffer_unit == "percent":
            # print(f"Expanding borders by {buffer_size}% for refinement order {ro}")
            new_polygon = buffer_polygon_by_percent(polygon["wkt"], buffer_size)
        elif buffer_unit == "km":
            # print(f"Expanding borders by {buffer_size}km for refinement order {ro}")
            new_polygon = buffer_polygon_in_km(polygon["wkt"], buffer_size)
        polygon_copy["wkt"] = new_polygon
        polygon_copy["refinement_order"] = ro
        new_polygons.append(polygon_copy)
    
    return new_polygons

def flatten_polygon_structures(polygon_structures) -> list:
    flattened = []
    for polygon_s in polygon_structures:
        # In case of a global polygon, we don't need to flatten it
        if "wkt" not in polygon_s:
            flattened.append(polygon_s)
            continue

        poly = polygon_s["wkt"]

        if isinstance(poly, MultiPolygon):
            for poly in poly.geoms:
                p_copy = copy.copy(polygon_s)
                p_copy["wkt"] = poly
                flattened.append(p_copy)
        elif isinstance(poly, Polygon):
            flattened.append(polygon_s)
    return flattened

def polygon_structures_preprocess(polygon_structures, base_refinement_order=0):
    # Do the necessary reshaping and/or buffering of the polygons
    new_polygons = []
    for polygon in polygon_structures:
        refinement_type = polygon["refinement_type"]
        if refinement_type == "uniform":
            new_polygons.extend(apply_buffers(polygon, base_refinement_order))
        elif refinement_type == "block":
            polygon["wkt"] = convex_hull(wkt_loads(polygon["wkt"]))

    polygon_structures.extend(new_polygons)

    # Extract all polygons
    polygon_structures = flatten_polygon_structures(polygon_structures)
    
    # Sort the polygon structure by refinement order
    polygon_structures = sorted(polygon_structures, key=lambda x: x["refinement_order"])

    print(polygon_structures)
    return polygon_structures

e = 1e-5
def generate_initial_mesh(refinement_order, radius=1.0, center=(0.0, 0.0, 0.0)):
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