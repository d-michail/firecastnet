"""
Shapefile Processing for WKT Generation

This module contains functions for processing shapefiles from local files or URLs,
converting them to WKT format for inclusion in the unified CSV. Uses fiona's built-in
support for zip+http:// URLs to handle remote zip archives directly.
"""

from typing import List
import fiona
from shapely import concave_hull
from shapely.geometry import shape
import os
import shapely.wkt
import pandas as pd
import numpy as np
from .geometry_utils import reshape_wkt_coords
from .clustering import points_clustering


def create_clustered_continent_geometry(continent_wkts, eps=2.0, min_samples=5):
    """
    Create continent geometry using DBSCAN clustering to handle disconnected regions.
    
    This function is useful for continents that have disconnected regions or islands
    that should be represented as separate polygons within a MultiPolygon.
    
    Args:
        continent_wkts (list): List of shapely geometries for countries in the continent
        eps (float): DBSCAN epsilon parameter for clustering (in degrees). Default 2.0.
        min_samples (int): DBSCAN minimum samples parameter. Default 5.
        
    Returns:
        str: WKT representation of the continent geometry (Polygon or MultiPolygon)
    """
    # Extract all coordinate points from the geometries
    continent_wkt = shapely.geometry.MultiPolygon(continent_wkts)
    points = [coord for geom in continent_wkt.geoms for coord in geom.exterior.coords]
    points = [(lat, lon) for lon, lat in points[:-1]]  # Flip coordinates

    if not points:
        return None
    
    # Convert points to numpy array for DBSCAN
    points_array = np.array(points)
    
    # Perform DBSCAN clustering
    clusters = points_clustering(points_array, eps=eps, min_samples=min_samples)
    
    if not clusters:
        # Fallback to simple approach if no clusters found
        mp_geom = shapely.geometry.MultiPolygon(continent_wkts)
        wkt_geom = concave_hull(mp_geom, 0.4)
        return shapely.wkt.dumps(wkt_geom)
    
    # Create polygons for each cluster
    polygons = []
    for cluster_points in clusters.values():
        try:
            # Create concave hull for each cluster
            if len(cluster_points) >= 4:  # Need at least 4 points for a polygon
                cluster_poly = concave_hull(shapely.geometry.Polygon(cluster_points), 0.4)
                if cluster_poly.is_valid and not cluster_poly.is_empty:
                    polygons.append(cluster_poly)
        except Exception as e:
            print(f"Error creating polygon for cluster: {e}")
            continue
        
    polygons = [poly for poly in polygons if isinstance(poly, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)) and not poly.is_empty]
    
    # Create the final geometry
    if len(polygons) == 0:
        # Fallback to simple approach if no valid polygons created
        mp_geom = shapely.geometry.MultiPolygon(continent_wkts)
        wkt_geom = concave_hull(mp_geom, 0.4)
        return shapely.wkt.dumps(wkt_geom)
    elif len(polygons) == 1:
        # Single polygon
        final_geometry = polygons[0]
    else:
        # Multiple polygons - create MultiPolygon
        final_geometry = shapely.geometry.MultiPolygon(polygons)
    
    return shapely.wkt.dumps(final_geometry)


def extract_shapefile_features(shapefile_paths, use_clustering=True):
    """
    Extract features from a shapefile and convert to WKT format.

    For each feature, attempts to extract the region name from the feature's properties
    using common field names. Falls back to provided region_name or filename if no
    name is found in properties.
    
    Args:
        shapefile_paths (list): List of paths to the shapefiles
        use_clustering (bool): Whether to use DBSCAN clustering for continent geometries

    Returns:
        list: List of dictionaries containing WKT data for each feature

    Note:
        The function looks for region names in the following property fields (in order):
        NAME, ADMIN, REGION, STATE_NAME, ADMIN_NAME, AREA, and their lowercase variants.
        Each feature gets its own name and code based on its properties.
        
        When use_clustering=True, DBSCAN clustering is applied to continent geometries
        to handle disconnected regions (like islands) as separate polygons in a MultiPolygon.
    """

    wkt_data = []
    
    country_to_codes = {}
    codes_to_country = {}
    continent_countries_list = {}
    continent_code_to_name = {
        "AF": "Africa",
        "AS": "Asia",
        "EU": "Europe",
        "NA": "North America",
        "OC": "Oceania",
        "SA": "South America",
        "AN": "Antarctica"
    }
    for iso, continent in continent_code_to_name.items():
        continent_countries_list[iso] = []
    df = pd.read_csv('csv/continent-country.csv', encoding='utf-8', sep=',', quotechar='"', keep_default_na=False)
    for _, row in df.iterrows():
        codes_to_country[row['ISO3']] = row['name']
        country_to_codes[row['name']] = row['ISO3']
        continent_countries_list[row['C-ISO3']].append(row['ISO3'])
    print(f"\nProcessing shapefiles...{shapefile_paths[0]}")

    # Firstly open the shapefile of Admin0 that contains countries
    try:
        with fiona.open(shapefile_paths[0]) as src:
            for i, feature in enumerate(src):
                print(f"Processing country {i+1}/{len(src)}")
                geom = shape(feature['geometry'])
                wkt_geom = reshape_wkt_coords(geom)
                try:
                    wkt_geom = concave_hull(wkt_geom, 0.5)
                except Exception as e:
                    print(f"Concave hull error for feature {i+1}: {e}")
                if not isinstance(wkt_geom, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
                    print(f"Skipping feature {i+1} due to invalid geometry type: {type(wkt_geom)}")
                    continue
                wkt_str = shapely.wkt.dumps(wkt_geom)
                
                feature_name = feature['properties'].get("shapeName")
                feature_iso = country_to_codes.get(feature_name, "UNK")
                
                print(feature_name, feature_iso)
                
                wkt_data.append({
                    "SUBUNIT": feature_name if feature_name else "Unknown",
                    "SU_A3": feature_iso,
                    "NAME": feature_name if feature_name else "Unknown",
                    "WKT": wkt_str
                })
    except Exception as e:
        print(f"Error processing Admin0 shapefile: {e}")

    #! NOT WORKING
    # Create continent entries by aggregating countries
    # for continent_iso, country_isos in continent_countries_list.items():
    #     print(f"Processing continent {continent_iso} with countries: {country_isos}")
    #     continent_name = continent_code_to_name.get(continent_iso, "Unknown")
    #     continent_wkts = []
        
    #     for entry in wkt_data:
    #         if entry['SU_A3'] in country_isos:
    #             geom = shapely.wkt.loads(entry['WKT'])
    #             if isinstance(geom, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
    #                 continent_wkts.append(geom)
        
    #     if continent_wkts:
    #         if use_clustering:
    #             # Use DBSCAN clustering to handle disconnected regions
    #             continent_wkt_str = create_clustered_continent_geometry(continent_wkts)
    #         else:
    #             # Use simple approach without clustering
    #             mp_geom = shapely.geometry.MultiPolygon(continent_wkts)
    #             wkt_geom = concave_hull(mp_geom, 0.4)
    #             continent_wkt_str = shapely.wkt.dumps(wkt_geom)
            
    #         if continent_wkt_str:
    #             wkt_data.append({
    #                 "SUBUNIT": continent_name,
    #                 "SU_A3": continent_iso,
    #                 "NAME": continent_name,
    #                 "WKT": continent_wkt_str
    #             })
    #             print(f"Added continent: {continent_name} with ISO {continent_iso}")
    #         else:
    #             print(f"Failed to create geometry for continent: {continent_name}")
    #     else:
    #         print(f"No countries found for continent: {continent_name}")


    # Now open the shapefile of Admin1 that contains states/provinces
    try:
        with fiona.open(shapefile_paths[1]) as src:
            for i, feature in enumerate(src):
                print(f"Processing state/province {i+1}/{len(src)}")
                geom = shape(feature['geometry'])
                wkt_geom = reshape_wkt_coords(geom)
                try:
                    wkt_geom = concave_hull(wkt_geom, 0.4)
                except Exception as e:
                    print(f"Concave hull error for feature {i+1}: {e}")
                wkt_str = shapely.wkt.dumps(wkt_geom)

                feature_name = feature['properties'].get("shapeName")
                admin0_name = feature['properties'].get("shapeGroup")
                feature_iso = country_to_codes.get(admin0_name, "UNK")
                feature_full_name = f"{feature_name}, {admin0_name}" if admin0_name else feature_name

                print(feature_full_name, feature_iso)

                wkt_data.append({
                    "SUBUNIT": feature_full_name if feature_full_name else "Unknown",
                    "SU_A3": feature_iso,
                    "NAME": feature_full_name if feature_full_name else "Unknown",
                    "WKT": wkt_str
                })
    except Exception as e:
        print(f"Error processing Admin1 shapefile: {e}")
    return wkt_data

def process_shapefiles(use_clustering=True) -> List[dict]:
    """
    Process shapefiles and return WKT data.
    
    Args:
        use_clustering (bool): Whether to use DBSCAN clustering for continent geometries
        
    Returns:
        List[dict]: List of dictionaries containing WKT data
    """
    # Check if required zip files exist
    sources = ['geoBoundariesCGAZ_ADM0.zip', 'geoBoundariesCGAZ_ADM1.zip']
    for i, file in enumerate(sources):
        if not os.path.exists("csv/" + file):
            sources[i] = f"zip+https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/CGAZ/{file}"
        else:
            # Fiona can't handle relative paths of a zip file, so convert to absolute path like that
            sources[i] = f"zip+file://{os.path.abspath('csv/' + file)}"
    all_wkt_data = []
    
    try:
        all_wkt_data.extend(extract_shapefile_features(sources, use_clustering=use_clustering))
    except Exception as e:
        print(f"Error processing shapefiles: {e}")
    
    return all_wkt_data