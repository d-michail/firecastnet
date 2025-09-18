"""
GFED Region Processing

This module contains functions for processing GFED (Global Fire Emissions Database)
regions from NetCDF data and generating WKT representations.
"""

import re
import numpy as np
import xarray as xr
import shapely
import shapely.wkt
from shapely.geometry import Polygon, MultiPolygon
from shapely import concave_hull

from .clustering import points_clustering

cube_zarr_path = "../../cube_v3.zarr"

def extract_gfed_regions():
    """
    Extract GFED region information from the dataset.
    
    Returns:
        list: List of tuples containing (region_id, region_name) for all regions except OCEAN
    """
    gfed_regions = xr.open_dataset(cube_zarr_path)["gfed_region"]
    matches = re.findall(r'(\d+)-([A-Z]+)', gfed_regions.description)
    # Extract region id and name and Remove the OCEAN region (0)
    return [(int(idx), name) for idx, name in matches][1:]


def extract_gfed_coordinates():
    """
    Extract coordinate points for each GFED region from the NetCDF dataset.
    
    Returns:
        dict: Dictionary mapping region IDs to lists of (lat, lon) coordinate tuples
    """
    gfed_regions = xr.open_dataset(cube_zarr_path)["gfed_region"]
    gfed_areas = extract_gfed_regions()
    print(gfed_areas)
    # Dataset resolution
    resolution = 0.25  # degrees
    total_lats = int((180 / resolution))
    
    # A map to hold the gfed region coordinate points
    gfed_map = {}
    for num, _ in gfed_areas:
        gfed_map[num] = []

    # Extract the gfed_region at every latitude index
    for i in range(total_lats):
        # Filter out the OCEAN region (0) too
        gfed_i = gfed_regions[i].where(gfed_regions[i] != 0, drop=True)
        lat = float(gfed_i.latitude.values)
        gfed_lons = gfed_i.longitude.values.tolist()
        for k, gfed_region in enumerate(gfed_i.values.astype(int)):
            lon = gfed_lons[k]
            gfed_map[gfed_region].append((lat, lon))
    
    return gfed_map


def create_gfed_geometry_with_clustering(points, eps=0.5, min_samples=3):
    """
    Create GFED region geometry using DBSCAN clustering.
    
    This function uses DBSCAN to cluster points into separate regions that can handle
    disconnected areas like island countries (e.g., New Zealand).
    
    Args:
        points (list): List of (lat, lon) coordinate tuples
        eps (float): DBSCAN epsilon parameter for clustering
        min_samples (int): DBSCAN minimum samples parameter
        
    Returns:
        str: WKT representation of the geometry (Polygon or MultiPolygon)
    """
    # Convert points to numpy array for DBSCAN
    points_array = np.array(points)
    
    # Perform DBSCAN clustering
    clusters = points_clustering(points_array, eps=eps, min_samples=min_samples)
    
    # Create polygons for each cluster
    polygons = []
    for cluster_points in clusters.values():
        # Create concave hull for each cluster
        cluster_poly = concave_hull(Polygon(cluster_points), 0.4)
        if cluster_poly.is_valid and not cluster_poly.is_empty:
            polygons.append(cluster_poly)
    
    # Create the final geometry
    if len(polygons) == 0:
        return None
    elif len(polygons) == 1:
        # Single polygon
        final_geometry = polygons[0]
    else:
        # Multiple polygons - create MultiPolygon
        final_geometry = MultiPolygon(polygons)
    
    # Store the WKT representation
    return shapely.wkt.dumps(final_geometry)


def create_gfed_geometry_simple(points):
    """
    Create GFED region geometry using simple concave hull without clustering.
    
    Args:
        points (list): List of (lat, lon) coordinate tuples
        
    Returns:
        str: WKT representation of the geometry
    """
    poly = concave_hull(Polygon(points), 0.4)
    return shapely.wkt.dumps(poly)


def process_gfed_regions(use_clustering=True, eps=0.5, min_samples=3):
    """
    Process GFED regions and convert them to WKT geometries.
    
    Args:
        use_clustering (bool): Whether to use DBSCAN clustering for disconnected regions
        eps (float): DBSCAN epsilon parameter
        min_samples (int): DBSCAN minimum samples parameter
        
    Returns:
        list: List of dictionaries containing GFED region data with WKT geometries
    """
    print("\nProcessing GFED regions...")
    
    gfed_areas = extract_gfed_regions()
    gfed_map = extract_gfed_coordinates()
    
    print("\nTransforming GFED regions to WKT geometries...")

    # Transform the points into WKT geometries
    gfed_data = []
    for region_idx, region_name in gfed_areas:
        points = gfed_map.get(region_idx, [])
        
        if not points:
            print(f"Warning: No points found for region {region_idx}")
            continue
            
        if use_clustering:
            region_wkt = create_gfed_geometry_with_clustering(points, eps, min_samples)
            if region_wkt is None:
                print(f"Warning: No valid polygons created for region {region_idx}")
                continue
        else:
            region_wkt = create_gfed_geometry_simple(points)
        
        gfed_data.append({
            "SUBUNIT": region_name,
            "SU_A3": region_name,
            "NAME": region_name,
            "WKT": region_wkt
        })

    return gfed_data