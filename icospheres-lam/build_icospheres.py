import copy
import json
import traceback
import yaml
import pymesh
import numpy as np
import pandas as pd
from shapely import convex_hull
from shapely.geometry import Polygon
from shapely.wkt import loads as wkt_loads
from buffers import buffer_polygon_by_percent, buffer_polygon_in_km
from utils import get_icosahedron_geometry, polygon_wraps_around_antimeridian, to_lat_lon, to_sphere


DEFAULT_BUFFER_FACTOR = 50.0
DEFAULT_REFINEMENT_TYPE = "none"
DEFAULT_BUFFER_UNIT = "km"

def argparse_setup():
    import argparse
    parser = argparse.ArgumentParser(description="Generate an icosphere with adaptive mesh refinement.")
    parser.add_argument(
        "--config", type=str, default="./config.yaml",
        help="Path to the configuration file."
    )
    return vars(parser.parse_args())

def find_intersecting_icosphere_faces(
    icosphere_vertices_latlon, icosphere_faces, spherical_polygon
) -> tuple:
    wraps_around = spherical_polygon["wraps_around"] if "wraps_around" in spherical_polygon else False
    intersections_idx = []
    polygon = wkt_loads(spherical_polygon["wkt"])
        
    # For each face of the icosphere, check if it intersects/contains the polygon
    for face_idx, face in enumerate(icosphere_faces):
        triangle_coords = [icosphere_vertices_latlon[k] for k in face]
        triangle = Polygon(triangle_coords)
        if polygon.intersects(triangle) or polygon.contains(triangle):
            if not wraps_around and polygon_wraps_around_antimeridian(triangle_coords):
                continue
            intersections_idx.append(face_idx)
    return intersections_idx

# Vertice indices map
# mts: mesh to submesh
# stm: submesh to mesh
def construct_submesh(mesh, intersecting_faces, submesh_refinement_order = 1):
    # Form a new isolated sub mesh from the intersecting faces and their vertices
    submesh = pymesh.form_mesh(mesh.vertices[np.unique(intersecting_faces)], intersecting_faces)
    
    # Create a 2-way dictionary to map the indices of the vertices (for later stitching) 
    original_vertice_indices = np.sort(np.unique(intersecting_faces)).tolist()
    vertice_maps = {
        "mts": {old_idx: new_idx for new_idx, old_idx in enumerate(original_vertice_indices)},
        "stm": {new_idx: old_idx for new_idx, old_idx in enumerate(original_vertice_indices)}
    }
    
    # Offset vertice index so that they start from 0 because the subdivision won't work otherwise
    new_intersecting_faces = np.array([[vertice_maps["mts"][old_idx] for old_idx in face] for face in intersecting_faces])
    submesh = pymesh.form_mesh(submesh.vertices, new_intersecting_faces)
    submesh = pymesh.subdivide(submesh, submesh_refinement_order)
    
    return submesh, vertice_maps

def mesh_stitch(mesh, submesh, intersecting_faces_idx, vertice_maps):
    # Remove the faces of the submesh from the original mesh
    new_mesh_faces = np.delete(mesh.faces, intersecting_faces_idx, axis=0)
    # Get the total count of vertices from the original mesh to remap the vertices of the new faces 
    total_vertice_count = np.max(mesh.faces) + 1
    
    # Update the faces of the submesh to be the original mesh vertices
    submesh_faces = []
    for face in submesh.faces:
        face_arr = []
        for vert_idx in face:
            # If the vertex is in the original mesh, use its original index
            # Otherwise, use the new index from the submesh
            if vert_idx in vertice_maps["stm"]:
                value = vertice_maps["stm"][vert_idx]
            else:
                value = vert_idx - len(vertice_maps["stm"]) + total_vertice_count
            face_arr.append(value)
        submesh_faces.append(face_arr)
    submesh_faces = np.array(submesh_faces, dtype=int)
    submesh_vertices = submesh.vertices[len(vertice_maps["stm"]):]
    
    # "Stitch" the updated submesh values to the original mesh
    new_mesh_faces = np.append(new_mesh_faces, submesh_faces, axis=0)
    new_mesh_vertices = np.append(mesh.vertices, submesh_vertices, axis=0)
    return pymesh.form_mesh(new_mesh_vertices, new_mesh_faces)

def generate_icosphere(polygon_structures, refinement_order, center, radius, save_to_file=True):
    """
    Generates a structured icosphere with adaptive mesh refinement based on polygon regions.

    Creates an icosphere mesh starting from a regular icosahedron, applies global subdivision,
    then performs selective refinement on faces intersecting with specified polygon regions.
    The process iteratively subdivides intersecting faces while maintaining mesh connectivity.
    
    Args:
        polygon_structures (list): List of dictionaries containing polygon definitions with keys:
            - 'wkt': Well-Known Text geometry string
            - 'refinement_order': Target subdivision level for this region
            - 'refinement_type': Type of refinement ('simple', 'uniform', 'block')
            - 'buffer_factor': Factor for buffer expansion (default 50.0)
            - 'buffer_unit': Unit for buffer expansion ('km' or 'percent')
        refinement_order (int): Base subdivision level applied to the entire icosphere
        center (tuple): 3D center coordinates for the sphere (x, y, z)
        radius (float): Radius of the generated sphere
        save_to_file (bool): Whether to save the result as 'icosphere.json'

    Returns:
        None: Function saves output to file and prints progress information

    Raises:
        Exception: For mesh generation errors or invalid polygon geometries

    Note:
        - Polygon structures are processed in order of increasing refinement_order
        - Face intersection uses lat/lon coordinate transformation
        - Handles antimeridian-crossing polygons with 'wraps_around' flag
        - Output includes vertices, faces, and face centroids in JSON format
    """
    # Get the initial icosahedron vertices and faces
    vertices, faces = get_icosahedron_geometry()

    # Initial mesh generation and subdivision
    mesh = pymesh.form_mesh(vertices, faces)
    mesh = pymesh.subdivide(mesh, refinement_order)
    mesh = pymesh.form_mesh(to_sphere(mesh.vertices, radius=radius, center=center), mesh.faces)
    
    try:
        cur_mesh = mesh
        if len(polygon_structures) != 0:
            # Sort the polygon structure by refinement order
            polygon_structures = sorted(polygon_structures, key=lambda x: x["refinement_order"])
            # Keep track of the current mesh and refinement order
            cur_refinement_order = refinement_order
            max_refinement_order = max(polygon["refinement_order"] for polygon in polygon_structures)
            while max_refinement_order >= cur_refinement_order:
                print("Current refinement order:", cur_refinement_order)
                intersecting_faces_idx = []
                for polygon in polygon_structures:
                    if polygon["refinement_order"] < cur_refinement_order:
                        continue
                    # Find the intersecting faces of the icosphere with the polygon
                    intersecting_faces_idx.extend(find_intersecting_icosphere_faces(
                        to_lat_lon(cur_mesh.vertices), cur_mesh.faces.tolist(), polygon
                    ))
                intersecting_faces_idx = np.unique(intersecting_faces_idx)
                intersecting_faces = np.array(cur_mesh.faces)[intersecting_faces_idx]

                # Subdivide the intersecting faces to create a submesh
                submesh, vertice_maps = construct_submesh(cur_mesh, intersecting_faces, 1)
                
                # Perform the stitching of the submesh to the original mesh
                cur_mesh = mesh_stitch(cur_mesh, submesh, intersecting_faces_idx, vertice_maps)

                # Conversion of mesh to spherical shape
                cur_mesh = pymesh.form_mesh(to_sphere(cur_mesh.vertices, radius=radius, center=center), cur_mesh.faces)

                # There is a better way to do this
                cur_refinement_order += 1

        # Save the final mesh to a json file
        if save_to_file:
            icospheres = {"vertices": [], "faces": []}
            icospheres["order_" + str(0) + "_vertices"] = cur_mesh.vertices
            icospheres["order_" + str(0) + "_faces"] = cur_mesh.faces
            cur_mesh.add_attribute("face_centroid")
            icospheres["order_" + str(0) + "_face_centroid"] = (
                cur_mesh.get_face_attribute("face_centroid")
            )
            icospheres_dict = {
                key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in icospheres.items()
            }
            with open("icosphere.json", "w") as f:
                json.dump(icospheres_dict, f)
    except Exception as e:
        print("An error occurred:", e)
        traceback.print_exc()

def apply_buffers(polygon):
    """
    Apply buffers to the polygon based on the refinement order and buffer factor.
    
    :param polygon: The polygon structure to apply buffers to.
    
    :return: A list of new polygons with applied buffers.
    """
    buffer_factor = polygon["buffer_factor"]
    buffer_unit = polygon["buffer_unit"]
    polygon_ro = polygon["refinement_order"]
    
    new_polygons = []
    if polygon["wkt"].startswith("MULTIPOLYGON"):
        for poly in wkt_loads(polygon["wkt"]).geoms:
            polygon_copy = copy.copy(polygon)
            polygon_copy["wkt"] = poly.wkt
            new_polygons.extend(apply_buffers(polygon_copy))
    else:
        for ro in range(refinement_order + 1, polygon_ro):
            polygon_copy = copy.copy(polygon)
            buffer_size = buffer_factor * (polygon_ro - ro)
            if buffer_unit == "percent":
                print(f"Expanding borders by {buffer_size}% for refinement order {ro}")
                new_poly_wkt = buffer_polygon_by_percent(polygon["wkt"], buffer_size).wkt
            elif buffer_unit == "km":
                print(f"Expanding borders by {buffer_size}km for refinement order {ro}")
                new_poly_wkt = buffer_polygon_in_km(polygon["wkt"], buffer_size).wkt
            polygon_copy["wkt"] = new_poly_wkt
            polygon_copy["refinement_order"] = ro
            new_polygons.append(polygon_copy)

    return new_polygons 

if __name__ == "__main__":    
    # Load the argument parser
    args = argparse_setup()

    # Load the config from the specified path
    config_path = args.get("config", "./config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load the countries csv
    csv_path = "./csv/wkt.csv"
    df = pd.read_csv(csv_path, encoding='latin1', sep=',', quotechar='"', keep_default_na=False)
        
    # Load the sphere from the config
    sphere_config = config["sphere"]
    radius = sphere_config.get("radius", 1.0)
    center = np.array(sphere_config.get("center", [0.0, 0.0, 0.0]))
    refinement_order = sphere_config.get("refinement_order", 3)
    
    # Load the refinement targets from the config
    refinement_targets = config.get("refinement_targets", [])
    polygon_structures = []
    for i, target in enumerate(refinement_targets):
        if "country_code" in target:
            target_wkt = df[df["SU_A3"] == target["country_code"]]["WKT"].values[0]
        elif "continent_code" in target:
            target_wkt = df[df["SU_A3"] == target["continent_code"]]["WKT"].values[0]
        elif "gfed_code" in target:
            target_wkt = df[df["SU_A3"] == target["gfed_code"]]["WKT"].values[0]
        elif "custom_wkt" in target:
            target_wkt = target["custom_wkt"]
        polygon_structures.append({
            "wkt": target_wkt,
            "refinement_order": target["refinement_order"],
            "refinement_type": target.get("refinement_type", DEFAULT_REFINEMENT_TYPE),
            "buffer_factor": target.get("buffer_factor", DEFAULT_BUFFER_FACTOR),
            "buffer_unit": target.get("buffer_unit", DEFAULT_BUFFER_UNIT)
        })

    # Do the necessary reshaping and/or buffering of the polygons
    new_polygons = []
    for polygon in polygon_structures:
        refinement_type = polygon["refinement_type"]
        if refinement_type == "uniform":
            new_polygons.extend(apply_buffers(polygon))
        elif refinement_type == "block":
            polygon["wkt"] = convex_hull(wkt_loads(polygon["wkt"])).wkt

    polygon_structures.extend(new_polygons)
    print("Polygon structure:", polygon_structures)
    
    generate_icosphere(polygon_structures, refinement_order, radius=radius, center=center)
