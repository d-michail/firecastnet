"""
Configuration handling and icosphere generation orchestration.

This module contains functions for processing configuration files,
initializing polygon structures from configuration data, and executing
the complete icosphere generation pipeline.
"""

import pandas as pd
from typing import List, Tuple, Dict, Any
from shapely.wkt import loads as wkt_loads

from .PolygonStructure import PolygonStructure
from .preprocess import generate_initial_mesh, polygon_structures_preprocess
from .utils import generate_icosphere_file_code
from .mesh_operations import generate_icosphere


def initialize_PolygonStructures(config: dict) -> Tuple[List[PolygonStructure], int]:
    """
    Initialize polygon structures from configuration data.
    
    Args:
        config (dict): Configuration dictionary containing sphere and refinement target settings
        
    Returns:
        Tuple[List[PolygonStructure], int]: Tuple containing list of polygon structures and refinement order
    """
    # Load the countries csv
    csv_path = "./csv/wkt.csv"
    df = pd.read_csv(csv_path, encoding='latin1', sep=',',
                     quotechar='"', keep_default_na=False)

    # Load the sphere from the config
    sphere_config = config["sphere"]
    refinement_order = sphere_config.get("refinement_order", 3)

    # Load the refinement targets from the config
    refinement_targets = config.get("refinement_targets", [])
    polygon_structures: List[PolygonStructure] = []
    for i, target in enumerate(refinement_targets):
        if "target_code" in target:
            code = target["target_code"]
            if "," in code:
                target_wkt = df[df["SUBUNIT"] == code]["WKT"].values[0]
            else:
                target_wkt = df[(df["SU_A3"] == code) | (df["SUBUNIT"] == code)]["WKT"].values[0]
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
        
    return polygon_structures, refinement_order


def execute_icosphere_generation(config: dict) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str]:
    """
    Execute the complete icosphere generation pipeline.
    
    Args:
        config (dict): Configuration dictionary containing all generation parameters
        
    Returns:
        Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str]: Tuple containing:
            - icospheres_dict: Main icosphere data
            - mesh_layers_dict: Mesh layers data (if enabled)
            - intersecting_mesh_layers_dict: Intersection layers data (if enabled)
            - file_code: Generated file code for naming
    """
    # Extract sphere parameters
    radius = config["sphere"].get("radius", 1.0)
    center = config["sphere"].get("center", [0.0, 0.0, 0.0])

    polygon_structures, refinement_order = initialize_PolygonStructures(config)
    file_code = generate_icosphere_file_code(polygon_structures, refinement_order)

    # Preprocess the polygon structures
    polygon_structures = polygon_structures_preprocess(polygon_structures, base_refinement_order=refinement_order)
    mesh = generate_initial_mesh(radius=radius, center=center)
    icospheres_dict, mesh_layers_dict, intersecting_mesh_layers_dict = generate_icosphere(
        polygon_structures,
        mesh,
        save_layers=config.get("save_layers", False),
        split_layers=config.get("split_layers", False),
        radius=radius,
        center=center
    )
    
    return icospheres_dict, mesh_layers_dict, intersecting_mesh_layers_dict, file_code