"""
Mesh operations for icosphere generation.

This module contains functions for mesh manipulation, face intersection detection,
submesh construction, and mesh stitching operations used in adaptive icosphere generation.
"""

try:
    import pymesh
except ImportError:
    Warning("pymesh is not installed. Please install it to generate icospheres.")
import numpy as np
import traceback
from typing import List, Dict, Any
from shapely.affinity import translate
from shapely.geometry import Polygon

from .utils import polygon_crosses_antimeridian, undo_antimeridian_wrap, to_lat_lon, to_sphere, mesh_to_dict
from .PolygonStructure import PolygonStructure


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

def generate_icosphere(polygon_structures: List[PolygonStructure], mesh, config: dict) -> Dict[str, Any]:
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
        config (dict): Configuration dictionary containing generation parameters such as
            sphere radius, center, and flags for saving mesh layers.

    Returns:
        Dict[str, Any]: Dictionary containing mesh data with keys:
            - 'order_0_vertices': Array of vertex coordinates
            - 'order_0_faces': Array of face indices  
            - 'order_0_face_centroid': Array of face centroid coordinates

    Note:
        - Processes polygons in order of increasing refinement_order
        - Handles antimeridian-crossing polygons with special logic
        - Reserved faces (interest=False) are excluded from subdivision
    """
    
    radius = config.get("sphere", {}).get("radius", 1.0)
    center = np.array(config.get("sphere", {}).get("center", [0.0, 0.0, 0.0]))
    all_layers = config.get("all_layers", False)
    split_layers = config.get("split_layers", False)

    def yield_polygons(ref_order: int, polygons: List[PolygonStructure]):
        for polygon in polygons:
            if polygon.refinement_order == ref_order:
                yield polygon
            elif polygon.interest == False and polygon.refinement_order <= ref_order:
                yield polygon


    mesh_layers = []
    split_layered_mesh = [None]
    try:
        if len(polygon_structures) != 0:
            max_ref_order = max(p.refinement_order for p in polygon_structures)
            
            last_ref_order = 0
            for ref_order in range(1, max_ref_order + 1):
                polygons = list(yield_polygons(ref_order, polygon_structures))
                if not polygons or polygons == [None]:
                    continue

                # Extract all unique faces of the icosphere that
                # intersect with the polygons of the current refinement order
                print("\n\nCurrent refinement order:", ref_order)
                intersecting_faces_idx = []
                reserved_faces = []
                # Make a copy of mesh and append it to mesh_layers
                if all_layers:
                    mesh_layers.append(pymesh.form_mesh(mesh.vertices, mesh.faces))

                for polygon in polygons:
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

                total_refinements = ref_order - last_ref_order
                # In case of global refinement there so need for selective refinement and stitching
                if len(intersecting_faces_idx) == len(mesh.faces) and \
                    len(reserved_faces) == 0:
                    mesh = pymesh.subdivide(mesh, total_refinements)
                    mesh = pymesh.form_mesh(to_sphere(mesh.vertices, radius=radius, center=center), mesh.faces)
                    split_layered_mesh.append(None)
                    last_ref_order = ref_order
                    continue
                
                intersecting_faces_idx = np.unique(intersecting_faces_idx)

                # Filter out reserved faces from the intersecting faces
                if len(reserved_faces) > 0:
                    intersecting_faces_idx = np.setdiff1d(intersecting_faces_idx, reserved_faces)
                intersecting_faces = np.array(mesh.faces)[intersecting_faces_idx]

                # Subdivide the intersecting faces to create a submesh
                submesh, vertice_maps = construct_submesh(mesh, intersecting_faces, total_refinements)

                # Perform the stitching of the submesh to the original mesh
                mesh = mesh_stitch(mesh, submesh, intersecting_faces_idx, vertice_maps)
                

                # Conversion of mesh to spherical shape
                mesh = pymesh.form_mesh(to_sphere(mesh.vertices, radius=radius, center=center), mesh.faces)

                last_ref_order = ref_order

                if split_layers:
                    split_layered_mesh.append(submesh)
                

        # Append the final mesh to the mesh layers
        mesh_layers.append(pymesh.form_mesh(mesh.vertices, mesh.faces))

        # Save the final mesh to a json file
        icospheres_dict = mesh_to_dict([mesh])
        mesh_layers_dict = mesh_to_dict(mesh_layers) if all_layers else None
        intersecting_faces_mesh_dict = mesh_to_dict(split_layered_mesh) if split_layers else None

        return icospheres_dict, mesh_layers_dict, intersecting_faces_mesh_dict

    except Exception as e:
        print("An error occurred:", e)
        traceback.print_exc()