import json
import pymesh
import traceback
import numpy as np
import pandas as pd
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.wkt import loads as wkt_loads
from preprocess import generate_initial_mesh, polygon_structures_preprocess
from utils import undo_antimeridian_wrap, generate_icosphere_file_code, load_yaml, polygon_crosses_antimeridian, to_lat_lon, to_sphere

DEFAULT_BUFFER_FACTOR = 50.0
DEFAULT_REFINEMENT_TYPE = "none"
DEFAULT_BUFFER_UNIT = "km"

def argparse_setup():
    import argparse
    parser = argparse.ArgumentParser(description="Generate an icosphere with adaptive mesh refinement.")
    parser.add_argument(
        "--config_path", 
        default="./config.yaml",
        dest="config_path",
        help="Path to the configuration file."
    )
    parser.add_argument(
        "--gzip", 
        action="store_true", 
        dest="gzip",
        default=False,
        help="Compress the output file with gzip.")
    parser.add_argument(
        "--output_dir",
        default="./icospheres/",
        dest="output_dir",
        help="Directory to save the generated icosphere files."
    )
    return vars(parser.parse_args())


def find_intersecting_icosphere_faces(
    icosphere_vertices_latlon: np.ndarray,
    icosphere_faces: list,
    polygon: Polygon,
) -> tuple:
    intersections_idx = []
    polygon_cross = polygon_crosses_antimeridian(polygon)
    # For each face of the icosphere, check if it intersects/contains the polygon
    for face_idx, face in enumerate(icosphere_faces):
        triangle = Polygon([icosphere_vertices_latlon[k] for k in face])
        triangle_cross = polygon_crosses_antimeridian(triangle)
        if not polygon_cross and not triangle_cross:
            if polygon.intersects(triangle) or polygon.contains(triangle):
                intersections_idx.append(face_idx)
        elif polygon_cross:
            if triangle_cross:
                triangle = undo_antimeridian_wrap(triangle)
            offset_triangle_left = translate(triangle, yoff=-360)
            offset_triangle_right = translate(triangle, yoff=360)
            if polygon.intersects(triangle) or polygon.contains(triangle):
                intersections_idx.append(face_idx)
            elif polygon.intersects(offset_triangle_left) or polygon.contains(offset_triangle_left):
                intersections_idx.append(face_idx)
            elif polygon.intersects(offset_triangle_right) or polygon.contains(offset_triangle_right):
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
    
def generate_icosphere(polygon_structures, mesh) -> dict:
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
        - Face intersection, submesh creation and stitching is done in lat/lon
        - Output includes vertices, faces, and face centroids in JSON format
    """
    
    def yield_polygons(ref_order, polygons):
        for polygon in polygons:
            if polygon["refinement_order"] == ref_order:
                yield polygon
            elif polygon["interest"] == False and polygon["refinement_order"] <= ref_order:
                yield polygon

    try:
        if len(polygon_structures) != 0:
            max_ref_order = max(p["refinement_order"] for p in polygon_structures)

            for ref_order in range(1, max_ref_order + 1):
                # Extract all unique faces of the icosphere that 
                # intersect with the polygons of the current refinement order
                print("\n\nCurrent refinement order:", ref_order)
                intersecting_faces_idx = []
                reserved_faces = []
                for polygon in yield_polygons(ref_order, polygon_structures):
                    print("Processing polygon target:", polygon["target_code"], 
                          "with refinement order:", polygon["refinement_order"],
                          "and interest:", polygon["interest"])

                    if polygon["target_code"] == "global":
                        intersecting_faces_idx = np.arange(len(mesh.faces))
                    else:
                        intersecting_faces = find_intersecting_icosphere_faces(
                            to_lat_lon(mesh.vertices), 
                            mesh.faces.tolist(),
                            polygon["wkt"]
                        )
                        if polygon["interest"] == False:
                            # When there is a target with negative interest then reserve/lock it's intersecting 
                            # faces so that they don't get subdivided
                            reserved_faces.extend(intersecting_faces)
                        else:
                            intersecting_faces_idx.extend(intersecting_faces)

                # In case of global refinement there so need for selective refinement and stitching
                if len(intersecting_faces_idx) == len(mesh.faces) and \
                    len(reserved_faces) == 0:
                    mesh = pymesh.subdivide(mesh, 1)
                    mesh = pymesh.form_mesh(to_sphere(mesh.vertices, radius=radius, center=center), mesh.faces)
                    continue
                
                intersecting_faces_idx = np.unique(intersecting_faces_idx)


                # If there are reserved faces, remove them from the intersecting faces
                if len(reserved_faces) > 0:
                    intersecting_faces_idx = np.setdiff1d(intersecting_faces_idx, reserved_faces)
                intersecting_faces = np.array(mesh.faces)[intersecting_faces_idx]

                # Subdivide the intersecting faces to create a submesh
                submesh, vertice_maps = construct_submesh(mesh, intersecting_faces, 1)
                
                # Perform the stitching of the submesh to the original mesh
                mesh = mesh_stitch(mesh, submesh, intersecting_faces_idx, vertice_maps)
                
                # Conversion of mesh to spherical shape
                mesh = pymesh.form_mesh(to_sphere(mesh.vertices, radius=radius, center=center), mesh.faces)
        
        # Save the final mesh to a json file
        icospheres = {"vertices": [], "faces": []}
        icospheres["order_" + str(0) + "_vertices"] = mesh.vertices
        icospheres["order_" + str(0) + "_faces"] = mesh.faces
        mesh.add_attribute("face_centroid")
        icospheres["order_" + str(0) + "_face_centroid"] = (
            mesh.get_face_attribute("face_centroid")
        )
        icospheres_dict = {
            key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in icospheres.items()
        }
        
        return icospheres_dict
    except Exception as e:
        print("An error occurred:", e)
        traceback.print_exc()


if __name__ == "__main__":    
    # Load the argument parser
    args = argparse_setup()

    # Load the config from the specified path
    config = load_yaml(args["config_path"])
    
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
        if "target_code" in target:
            code = target["target_code"]
            target_wkt = df[df["SU_A3"] == code]["WKT"].values[0]
        elif "custom_wkt" in target:
            code = "custom"
            target_wkt = target["custom_wkt"]
        else:
            raise ValueError(f"Target {i} does not have a valid 'target_code' or 'custom_wkt' field.")
        
        polygon_structures.append({
            "target_code": code,
            "wkt": wkt_loads(target_wkt),
            "refinement_order": target["refinement_order"],
            "refinement_type": target.get("refinement_type", DEFAULT_REFINEMENT_TYPE),
            "buffer_factor": target.get("buffer_factor", DEFAULT_BUFFER_FACTOR),
            "buffer_unit": target.get("buffer_unit", DEFAULT_BUFFER_UNIT),
            "interest": target.get("interest", True)
        })
    
    for i in range(1, refinement_order + 1):
        # Add a default polygon structure for the base refinement order
        polygon_structures.append({
            "target_code": "global",
            "refinement_type": DEFAULT_REFINEMENT_TYPE,
            "refinement_order": i,
            "interest": True,
        })
    
    file_code = generate_icosphere_file_code(polygon_structures, refinement_order)
    icosphere_location = args['output_dir'] + f"{file_code}.json"

    # Preprocess the polygon structures
    polygon_structures = polygon_structures_preprocess(polygon_structures, base_refinement_order=refinement_order)

    for poly in polygon_structures:
        if "wkt" in poly and polygon_crosses_antimeridian(poly["wkt"]):
            print(f"Polygon {poly['target_code']} with refinement order {poly['refinement_order']} crosses the antimeridian.")
    mesh = generate_initial_mesh(0, radius=radius, center=center)
    icospheres_dict = generate_icosphere(polygon_structures, mesh)
    
    # For debugging purposes, you can validate the mesh structure
    # validate_mesh(file_code, icospheres_dict)

    # Save the icosphere to a file
    with open(icosphere_location, 'w') as f:
        json.dump(icospheres_dict, f)
        
    # Optionally gzip the file
    if args["gzip"] or config.get("gzip", False):
        from utils import gzip_file
        gzip_file(icosphere_location)

    print(f"Generated icosphere saved to {icosphere_location}")