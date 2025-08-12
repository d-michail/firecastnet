import json
import pymesh
import traceback
import numpy as np
import pandas as pd
from PolygonStructure import PolygonStructure
from typing import List, Dict, Any
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.wkt import loads as wkt_loads
from preprocess import generate_initial_mesh, polygon_structures_preprocess
from utils import undo_antimeridian_wrap, generate_icosphere_file_code, load_yaml, polygon_crosses_antimeridian, to_lat_lon, to_sphere, mesh_to_dict

def argparse_setup():
    import argparse
    parser = argparse.ArgumentParser(description="Generate an icosphere with adaptive mesh refinement.")
    parser.add_argument(
        "--config", 
        default="./config.yaml",
        dest="config_path",
        help="Path to the configuration file."
    )
    parser.add_argument(
        "--out_dir",
        default="./icospheres/",
        dest="output_dir",
        help="Directory to save the generated icosphere files."
    )
    return vars(parser.parse_args())


def find_intersecting_icosphere_faces(
    icosphere_vertices_latlon: np.ndarray,
    icosphere_faces: List[List[int]],
    polygon: Polygon,
) -> List[int]:
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
            # If the vertex is in the original mesh, use it's original index
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

def generate_icosphere(polygon_structures: List[PolygonStructure], mesh, save_layers: bool) -> Dict[str, Any]:
    """
    Generates an icosphere with adaptive mesh refinement based on polygon regions.
    
    Starting from an initial icosphere mesh, this function applies selective subdivision
    to faces that intersect with specified polygon regions. The refinement process
    iterates through increasing refinement orders, subdividing intersecting faces
    while maintaining mesh connectivity through submesh construction and stitching.
    
    Args:
        polygon_structures (List[PolygonStructure]): List of polygon structures defining
            refinement regions, each containing target geometry, refinement order,
            and refinement parameters.
        mesh: PyMesh mesh object representing the initial icosphere.
        save_layers (bool): Flag indicating whether to save intermediate mesh layers.

    Returns:
        Dict[str, Any]: Dictionary containing mesh data with keys:
            - 'order_0_vertices': Array of vertex coordinates
            - 'order_0_faces': Array of face indices  
            - 'order_0_face_centroid': Array of face centroid coordinates

    Note:
        - Uses global variables radius and center for sphere projection
        - Processes polygons in order of increasing refinement_order
        - Handles antimeridian-crossing polygons with special logic
        - Reserved faces (interest=False) are excluded from subdivision
    """

    def yield_polygons(ref_order: int, polygons: List[PolygonStructure]):
        for polygon in polygons:
            if polygon.refinement_order == ref_order:
                yield polygon
            elif polygon.interest == False and polygon.refinement_order <= ref_order:
                yield polygon


    mesh_layers = []
    intersecting_mesh_layers = [None]
    try:
        if len(polygon_structures) != 0:
            max_ref_order = max(p.refinement_order for p in polygon_structures)

            for ref_order in range(1, max_ref_order + 1):
                # Extract all unique faces of the icosphere that 
                # intersect with the polygons of the current refinement order
                print("\n\nCurrent refinement order:", ref_order)
                intersecting_faces_idx = []
                reserved_faces = []
                # Make a copy of mesh and append it to mesh_layers
                if save_layers:
                    mesh_layers.append(pymesh.form_mesh(mesh.vertices, mesh.faces))
 
                for polygon in yield_polygons(ref_order, polygon_structures):
                    print("Processing polygon target:", polygon.target_code, 
                          "with refinement order:", polygon.refinement_order,
                          "and interest:", polygon.interest)

                    if polygon.target_code == "global":
                        intersecting_faces_idx = np.arange(len(mesh.faces))
                    else:
                        intersecting_faces = find_intersecting_icosphere_faces(
                            to_lat_lon(mesh.vertices), 
                            mesh.faces.tolist(),
                            polygon.wkt
                        )
                        if polygon.interest == False:
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
                    intersecting_mesh_layers.append(None)
                    continue
                
                intersecting_faces_idx = np.unique(intersecting_faces_idx)

                # If there are reserved faces, remove them from the intersecting faces
                if len(reserved_faces) > 0:
                    intersecting_faces_idx = np.setdiff1d(intersecting_faces_idx, reserved_faces)
                intersecting_faces = np.array(mesh.faces)[intersecting_faces_idx]

                intersecting_mesh_layers.append(pymesh.form_mesh(mesh.vertices, intersecting_faces))

                # Subdivide the intersecting faces to create a submesh
                submesh, vertice_maps = construct_submesh(mesh, intersecting_faces, 1)
                
                # Perform the stitching of the submesh to the original mesh
                mesh = mesh_stitch(mesh, submesh, intersecting_faces_idx, vertice_maps)
                
                # Conversion of mesh to spherical shape
                mesh = pymesh.form_mesh(to_sphere(mesh.vertices, radius=radius, center=center), mesh.faces)

        # Append the final mesh to the mesh layers
        mesh_layers.append(pymesh.form_mesh(mesh.vertices, mesh.faces))

        # Save the final mesh to a json file
        icospheres_dict = mesh_to_dict([mesh])
        mesh_layers_dict = mesh_to_dict(mesh_layers) if save_layers else None
        intersecting_faces_mesh_dict = mesh_to_dict(intersecting_mesh_layers) if intersecting_mesh_layers else None

        return icospheres_dict, mesh_layers_dict, intersecting_faces_mesh_dict

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
    polygon_structures: List[PolygonStructure] = []
    for i, target in enumerate(refinement_targets):
        if "target_code" in target:
            code = target["target_code"]
            target_wkt = df[df["SU_A3"] == code]["WKT"].values[0]
        elif "custom_wkt" in target:
            code = "custom"
            target_wkt = target["custom_wkt"]
        else:
            raise ValueError(f"Target {i} does not have a valid 'target_code' or 'custom_wkt' field.")
        target["wkt"] = wkt_loads(target_wkt)
        polygon_structures.append(PolygonStructure.from_dict(target))

    for i in range(1, refinement_order + 1):
        # Add a default polygon structure for the base refinement order
        polygon_structures.append(PolygonStructure(
            target_code="global",
            refinement_order=i,
        ))

    file_code = generate_icosphere_file_code(polygon_structures, refinement_order)
    icosphere_location = args['output_dir'] + f"{file_code}.json"

    # Preprocess the polygon structures
    polygon_structures = polygon_structures_preprocess(polygon_structures, base_refinement_order=refinement_order)
    mesh = generate_initial_mesh(radius=radius, center=center)
    icospheres_dict, mesh_layers_dict, intersecting_faces_mesh_dict = generate_icosphere(polygon_structures, mesh, save_layers=config.get("save_layers", False))

    # Save the icosphere to a file
    with open(icosphere_location, 'w') as f:
        json.dump(icospheres_dict, f)
    
    # Save mesh layers if they were generated
    if mesh_layers_dict:
        layers_location = args['output_dir'] + f"{file_code}_layers.json"
        with open(layers_location, 'w') as f:
            json.dump(mesh_layers_dict, f)
        print(f"Generated mesh layers saved to {layers_location}")

    if intersecting_faces_mesh_dict:
        faces_location = args['output_dir'] + f"{file_code}_intersecting_faces.json"
        with open(faces_location, 'w') as f:
            json.dump(intersecting_faces_mesh_dict, f)
        print(f"Generated intersecting faces saved to {faces_location}")

    # Optionally gzip the file
    if config.get("gzip", False):
        from utils import gzip_file
        gzip_file(icosphere_location)
        if mesh_layers_dict:
            gzip_file(layers_location)
        if intersecting_faces_mesh_dict:
            gzip_file(faces_location)

    print(f"Generated icosphere saved to {icosphere_location}")