import json
from typing import List
import numpy as np
import gzip

from visuals.icosphere import visualize_icosphere_latlon_2d, visualize_icosphere_3d, visualize_icosphere_layered_3d, visualize_icosphere_layered_latlon_2d, visualize_multiple_meshes_3d, visualize_multiple_meshes_latlon_2d, visualize_wkt
from utils import generate_icosphere_file_code, load_yaml, order
from PolygonStructure import PolygonStructure

sp_res = 0.250
embed_cube = False
max_lat = 89.5
min_lat = -89.5
max_lon = 179.5
min_lon = -179.5

local_dir = "icospheres-lam"

def generate_icosphere_file_code(polygon_structures: List[PolygonStructure], ref_order: int, mesh_layers: bool = False, intersecting_faces: bool = False) -> str:
    structure_code = "icosphere_s"+str(ref_order)
    for p in polygon_structures:
        if p.target_code == "global":
            continue
        if not p.refinement_type:
            p.refinement_type = "none"
        structure_code += f"_{p.target_code}_{p.refinement_order}{p.refinement_type[0]}"
    
    if mesh_layers:
        structure_code += "_layers"
    # if intersecting_faces:
    #     structure_code += "_intersecting_faces"

    return structure_code

def mesh_to_np(mesh):
    o, _ = get_minmax_order(mesh)

    while order(o, "vertices") in mesh:
        # Convert to numpy arrays
        vertices = np.array(mesh[order(o, "vertices")])
        faces = np.array(mesh[order(o, "faces")])
        face_centroids = np.array(mesh[order(o, "face_centroid")])
        
        # Find unique vertices referenced in faces
        used_vertices = np.unique(faces.flatten())
        
        # Create mapping from old to new vertex indices
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
        
        # Filter vertices to only keep used ones
        filtered_vertices = vertices[used_vertices]
        
        # Remap face indices to new vertex indices
        remapped_faces = np.array([[vertex_map[v] for v in face] for face in faces])
        
        # Update mesh with filtered data
        mesh[order(o, "vertices")] = filtered_vertices
        mesh[order(o, "faces")] = remapped_faces
        mesh[order(o, "face_centroid")] = face_centroids
        o += 1


def get_minmax_order(icosphere):
    min_order = 0
    while order(min_order, "vertices") not in icosphere:
        min_order += 1
    max_order = min_order
    while order(max_order, "vertices") in icosphere:
        max_order += 1
    return min_order, max_order

if __name__ == "__main__":
    config = load_yaml(local_dir + "/config.yaml")
    save_layers = config.get("save_layers", False)

    # Load the polygon structures from the config
    icosphere_structs = [PolygonStructure.from_dict(p) for p in config["refinement_targets"]]

    # Generate the icosphere file code and read the file contents of the mesh
    icosphere_file_code = generate_icosphere_file_code(
        icosphere_structs,
        config["sphere"]["refinement_order"],
        mesh_layers=save_layers,
        intersecting_faces=True
    )
    file_path = f"{local_dir}/icospheres/{icosphere_file_code}.json"
    if config.get("gzip", False):
        file_path += ".gz"
    
    with (gzip.open(file_path, 'rt') if config.get("gzip", False) else open(file_path, 'r')) as f:
        icosphere = json.load(f)

    # Convert the mesh to numpy arrays for easier manipulation
    mesh_to_np(icosphere)

    min_order, max_order = get_minmax_order(icosphere)
    total_orders = max_order - min_order
    
    # Visualization settings
    show = True
    show_3d = True
    show_2d = True

    if config.get("save_layers", False):
        visualize_icosphere_layered_3d(
            icosphere, 
            max_columns=4, 
            colors=["royalblue"] * max_order,
            rotation=(-26.5, -57.5, 0),
            alpha=1.0,
            zoom=[1.15, 1.3, 1.5],
            title=f"Icosphere Layers - SHSA GFED Region",
        )
    else:
        # Visualize each refinement level separately
        # if show_3d:
        #     visualize_icosphere_3d(icosphere, refinement_order=0)
        if show_2d:
            visualize_icosphere_layered_latlon_2d(
                icosphere,
                max_columns=2,
                colors=["#e16b43"] * max_order,
                show_wireframe=True,
                alpha=1.0,
                title="Icosphere Layers - World Map View - SHSA GFED Region"
            )
        labels = [f"Refinement Order {o}" for o in range(min_order, max_order)]
        for o in range(min_order, max_order):
           buffer_size = (o - total_orders - 1) * icosphere_structs[0].buffer_factor
           if buffer_size > 0.1:
               labels[total_orders - o] += f" (Buffer Size: {buffer_size:.2f} km)"

        visualize_multiple_meshes_latlon_2d(
            meshes=[
            {"vertices": icosphere[order(o, "vertices")], "faces": icosphere[order(o, "faces")]}
            for o in range(min_order, max_order)
            ],
            colors=["#da9881", "#e16b43", "#e2420b"],
            labels=labels,
            alpha=1.0,
            title="Icosphere Layers - World Map View - SHSA GFED Region",
        )