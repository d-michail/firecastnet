#!/usr/bin/env python3

import json
from typing import List
import gzip
import os

from visuals import d2, d3
from icosphere_generation.utils import load_yaml, order
from icosphere_generation.PolygonStructure import PolygonStructure

def generate_icosphere_file_code(polygon_structures: List[PolygonStructure], ref_order: int, \
                mesh_layers: bool = False, split_layers: bool = False) -> str:
    structure_code = "icosphere_s"+str(ref_order)
    for p in polygon_structures:
        if p.target_code == "global":
            continue
        if not p.refinement_type:
            p.refinement_type = "none"
        structure_code += f"_{p.target_code}_{p.refinement_order}{p.refinement_type[0]}"
    
    if mesh_layers:
        structure_code += "_all_layers"
    elif split_layers:
        structure_code += "_split_layers"

    return structure_code

def mesh_to_np(mesh):
    import numpy as np
    min_order, _ = get_minmax_order(mesh)

    while order(min_order, "vertices") in mesh:
        # Convert to numpy arrays
        vertices = np.array(mesh[order(min_order, "vertices")])
        faces = np.array(mesh[order(min_order, "faces")])
        face_centroids = np.array(mesh[order(min_order, "face_centroid")])
        
        # Find unique vertices referenced in faces
        used_vertices = np.unique(faces.flatten())
        
        # Create mapping from old to new vertex indices
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
        
        # Filter vertices to only keep used ones
        filtered_vertices = vertices[used_vertices]
        
        # Remap face indices to new vertex indices
        remapped_faces = np.array([[vertex_map[v] for v in face] for face in faces])
        
        # Update mesh with filtered data
        mesh[order(min_order, "vertices")] = filtered_vertices
        mesh[order(min_order, "faces")] = remapped_faces
        mesh[order(min_order, "face_centroid")] = face_centroids
        min_order += 1


def get_minmax_order(icosphere:dict):
    vertex_keys = [int(key[6]) for key in icosphere.keys() if "_vertices" in key]
    if len(vertex_keys) == 0:
        raise ValueError("No order_N_vertices field found in icospheres file")
    sorted_orders = sorted(vertex_keys)
    return sorted_orders[0], sorted_orders[-1]

def arg_parse():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize generated icospheres.")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--icosphere-dir",
        default="./icospheres/",
        dest="icosphere_dir",
        help="Directory containing the icosphere files."
    )    
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    config = load_yaml(args.config)
    all_layers = config.get("all_layers", False)
    split_layers = config.get("split_layers", False)

    # Load the polygon structures from the config
    icosphere_structs = [PolygonStructure.from_dict(p) for p in config["refinement_targets"]]

    # Generate the icosphere file code and read the file contents of the mesh
    icosphere_file_code = generate_icosphere_file_code(
        icosphere_structs,
        config["sphere"]["refinement_order"],
        mesh_layers=all_layers,
        split_layers=split_layers
    )
    file_path = os.path.join(args.icosphere_dir, f"{icosphere_file_code}.json")
    if config.get("gzip", False):
        file_path += ".gz"
    
    with (gzip.open(file_path, 'rt') if config.get("gzip", False) else open(file_path, 'r')) as f:
        icosphere = json.load(f)

    # Convert the mesh to numpy arrays for easier manipulation
    mesh_to_np(icosphere)

    min_order, max_order = get_minmax_order(icosphere)
    total_orders = max_order - min_order + 1
    
    if config.get("all_layers", False):
        # Visualize each refinement level separately
        d3.visualize_icosphere_layered_3d(
            icosphere, 
            max_columns=4, 
            colors=["#e16b43"] * (max_order + 1),
            rotation=(-26.5, -57.5, 0),
            alpha=0.825,
            zoom=[1, 1, 1, 1, 1.15, 1.3, 1.45],
            title=f"Icosphere Mesh Progression - SHSA GFED Region",
        )
    elif config.get("split_layers", False):
        # Visualize each refinement level separately
        labels = [f"Refinement Order {o}" for o in range(min_order, max_order + 1)]
        print(labels)
        for o in range(min_order, max_order + 1):
            print(o, o - total_orders - 1)
            buffer_size = (o - total_orders - 1) * icosphere_structs[0].buffer_factor
            if buffer_size > 0.1:
                labels[o - total_orders - 1] += f" (Buffer Size: {buffer_size:.2f} km)"
        if icosphere_structs[0].refinement_type == "uniform":
            d2.visualize_icosphere_layered_latlon_2d(
                icosphere,
                max_columns=2,
                colors=["#da9881", "#e16b43", "#e2420b"],
                show_wireframe=True,
                alpha=1.0,
                title="Icosphere Target Layers - SHSA GFED Region",
                labels=labels,
                show_title=False,
                crop_bounds={"lat": [-70, 15], "lon": [-120, 0]},
                # save_path="../thesis/figures/icosphere_target_layers_latlon.png"
            )
        d2.visualize_multiple_meshes_latlon_2d(
            meshes=[
            {"vertices": icosphere[order(o, "vertices")], "faces": icosphere[order(o, "faces")]}
            for o in range(min_order, max_order + 1)
            ],
            colors=["#da9881", "#e16b43", "#e2420b"],
            labels=labels,
            alpha=1.0,
            # title="Icosphere Layers - World Map View - SHSA GFED Region",
            show_title=False,
            show_labels=False,
            crop_bounds={"lat": [-80, 20], "lon": [-110, -10]},
            # save_path="../thesis/figures/icosphere_target_layers_stacked_latlon.png"
        )
    else:
        # Visualize the entire icosphere mesh in 3D and 2D
        d3.visualize_icosphere_layered_3d(
            icosphere, 
            max_columns=1, 
            colors=["#da9881", "#e16b43", "#e2420b", "#e2420b", "#e2420b"],
            # rotation=(-26.5, -57.5, 0),
            alpha=1,
            # zoom=[1.3],
            # title=f"Icosphere Mesh 3D - SHSA GFED Region",
            show_title=False,
            show_grid=False,
            show_labels=False,
            save_path=f"./figures/{icosphere_file_code}_3d.png"
        )
        d2.visualize_icosphere_latlon_2d(
            icospheres=icosphere,
            refinement_order=0,
            show_wireframe=True,
            alpha=1,
            show_title=False,
        )

