import re
import xarray as xr
import pandas as pd
import numpy as np
import shapely
import shapely.wkt
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon, MultiPolygon
from shapely import concave_hull
from shapely.ops import transform as shapely_transform

csv_dir = 'icospheres-lam/csv'

def flip_coords(x, y):
    return y, x

def get_country(country_name, filename=f"{csv_dir}/wkt.csv"):
    df = pd.read_csv(filename, encoding='latin1', keep_default_na=False, sep=',', quotechar='"')
    return df[df['NAME'] == country_name or df['SU_A3'] == country_name or df['SUBUNIT'] == country_name].iloc[0]

def reshape_wkt_coords(wkt_shape):
    # Remove the last coordinate (which duplicates the first in shapely)
    # Flip the coordinates to (lat, lon) format.........
    # Create polygon with unique coordinates
    if isinstance(wkt_shape, Polygon):
        return shapely_transform(flip_coords, Polygon(wkt_shape))
    elif isinstance(wkt_shape, MultiPolygon):
        shapes = []
        for poly in wkt_shape.geoms:
            shapes.append(shapely_transform(flip_coords, Polygon(poly)))
        return MultiPolygon(shapes)
    else:
        raise ValueError("Shape must be a Polygon or MultiPolygon")

def points_clustering(points, eps=0.5, min_samples=3):
    """
    Clusters points using DBSCAN algorithm.

    Args:
        points (list of tuples): List of (lat, lon) tuples representing points.
        eps (float): The maximum distance between two samples for one to be considered
                     as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be
                           considered as a core point.

    Returns:
        dict: A dictionary where keys are cluster labels and values are lists of points in each cluster.
    """
    points_array = np.array(points)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(points_array)

    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label == -1:  # Skip noise points
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(points[i])
    
    return clusters

def format_countries_wkt_csv_file():
    """
    Generates a comprehensive CSV file containing WKT geometries for countries, continents, and GFED regions.

    This function processes multiple geographic data sources to create a unified CSV file with
    Well-Known Text (WKT) representations of geographic boundaries. It performs three main operations:
    
    1. Extracts and transforms country geometries from a Natural Earth dataset
    2. Aggregates country geometries into continent-level polygons using concave hulls
    3. Processes GFED (Global Fire Emissions Database) regions from NetCDF data, using DBSCAN
       clustering to handle disconnected regions (such as island nations) as MultiPolygons
    
    The function transforms all coordinates from (lon, lat) to (lat, lon) format for consistency
    and applies coordinate flipping where necessary to maintain proper geographic projections.
    
    Data Sources:
        - Countries: 110m-admin-0-countries.csv (Natural Earth dataset) (https://github.com/zzolo/geo_simple_countries_wkt_csv)
        - Continents: continent-country.csv (continent-country mapping)
        - GFED Regions: cube.zarr (NetCDF dataset with global fire emission regions)
    
    Processing Steps:
        1. **Countries**: Loads country WKT geometries, flips coordinates to (lat, lon) format
        2. **Continents**: Groups countries by continent, creates concave hulls for each continent
        3. **GFED Regions**: 
           - Extracts coordinate points from NetCDF grid data (0.25° resolution)
           - Uses DBSCAN clustering (eps=0.5°, min_samples=3) to separate disconnected regions
           - Creates individual polygons for each cluster and combines into MultiPolygons
           - Handles island nations and archipelagos as separate polygon components
    
    Output:
        Creates 'wkt_countries_continents.csv' with columns:
        - SUBUNIT: Administrative subunit name
        - SU_A3: ISO 3-letter country/region code
        - NAME: Display name
        - WKT: Well-Known Text geometry representation
    
    Raises:
        FileNotFoundError: If required input CSV or NetCDF files are not found
        ValueError: If WKT geometries cannot be parsed or are invalid
        
    Note:
        The function prints progress information during processing, including latitude
        index progress for GFED regions and warnings for regions where polygon
        creation fails.
    """
    use_dbscan = False # Set to false to skip DBSCAN clustering and use concave hulls directly
    countries_csv = f"{csv_dir}/110m-admin-0-countries.csv"
    continents_csv = f"{csv_dir}/continent-country.csv"
    
    # --------- WKT BY COUNTRIES ---------
    print("Copying wkt geometries of countries...")
    df = pd.read_csv(countries_csv, encoding='latin1', sep=';', quotechar='"')

    # Define the columns you want to extract
    columns_to_extract = ["SUBUNIT", "SU_A3", "NAME", 'WKT']  # Replace with your desired column names

    # Extract the specified columns
    df = df[columns_to_extract]
    wkt = df["WKT"]
    wkt = wkt.apply(lambda x: shapely.wkt.loads(x))
    wkt = wkt.apply(lambda x: reshape_wkt_coords(x))
    wkt = wkt.apply(lambda x: shapely.wkt.dumps(x))
    df["WKT"] = wkt
    wkt_df = df.copy()

    # --------- WKT BY CONTINENTS ---------
    print("\nProcessing continents...") 
    df = pd.read_csv(continents_csv, encoding='utf-8', sep=',', quotechar='"', keep_default_na=False) # OMG.....
    
    countries_dict = []
    # Locate and extract all the wkt geometries of all countries by continent
    for row in df.to_dict(orient='records'):
        c = row['continent']
        iso = row['C-ISO3']
        country_wkt = df[df['continent'] == c]['ISO3'].tolist()
        country_wkt = wkt_df[wkt_df['SU_A3'].isin(country_wkt)]['WKT']
        countries_dict.append({"wkt":country_wkt, "c-name": c, "c-iso": iso})
    
    continent_isos = df['C-ISO3'].unique().tolist()
    continents_dict = {}
    for iso in continent_isos:
        continents_dict[iso] = {}
        countries = df[df['C-ISO3'] == iso]
        countries_wkt = wkt_df[wkt_df['SU_A3'].isin(countries["ISO3"])]['WKT'].apply(lambda x: shapely.wkt.loads(x)).to_list()
        continent_name = countries.values[0][2]
        continent_wkts = []
        for wkt in countries_wkt:
            if isinstance(wkt, Polygon):
                continent_wkts.append(wkt)
            elif isinstance(wkt, MultiPolygon):
                continent_wkts.extend(wkt.geoms)

        # Initialize the continent dictionary with empty WKT list and name
        continents_dict[iso] = {"wkt": continent_wkts, "name": continent_name, "iso": iso}
    
    # "Flatten" the geometries of all countries by continent
    for iso in continents_dict:
        continent = continents_dict[iso]
        # Concatenate all the polygons into a single MultiPolygon and create a concave hull
        continent["wkt"] = concave_hull(MultiPolygon(continent["wkt"]), 0.4)
        # Store the WKT representation
        continent["wkt"] = shapely.wkt.dumps(continent["wkt"])

    # Add the continent column to the dataframe
    for iso in continent_isos:
        c = continents_dict[iso]
        wkt_df.loc[len(wkt_df)] = [c["name"], c["iso"], c["name"], c["wkt"]]

    # --------- WKT BY GFED REGIONS ---------
    print("\nProcessing GFED regions...")
    gfed_regions = xr.open_dataset("cube.zarr")["gfed_region"]

    matches = re.findall(r'(\d+)-([A-Z]+)', gfed_regions.description)
    # Extract region id and name and Remove the OCEAN region (0)
    gfed_areas = [(int(idx), name) for idx, name in matches][1:]

    # Dataset resolution
    resolution = 0.25  # degrees
    total_lats = int((180 / resolution))
    
    # A map to hold the gfed region coordinate points
    gfed_map = {}
    for num, _ in gfed_areas:
        gfed_map[num] = []
    
    # Extract the gfed_region at every latitude index
    for i in range(total_lats):
        if i % 100 == 0 or i == total_lats - 1:
            print(f"\tProcessing latitude index {i}/{total_lats - 1}")
        # Filter out the OCEAN region (0) too
        gfed_i = gfed_regions[i].where(gfed_regions[i] != 0, drop=True)
        lat = float(gfed_i.latitude.values)
        gfed_lons = gfed_i.longitude.values.tolist()
        for k, gfed_region in enumerate(gfed_i.values.astype(int)):
            lon = gfed_lons[k]
            gfed_map[gfed_region].append((lat, lon))
    
    print("\nTransforming GFED regions to WKT geometries...")

    # Transform the points into WKT geometries
    gfed_wkt_map = {}
    for region_idx, points in gfed_map.items():
        if use_dbscan: 
            poly = concave_hull(Polygon(points), 0.4)
            gfed_wkt_map[region_idx] = shapely.wkt.dumps(poly)
            continue
        # Use DBSCAN to cluster points into separate regions that contain Island countries (e.g., New Zealand)
        # Convert points to numpy array for DBSCAN
        points_array = np.array(points)
        
        # Perform DBSCAN clustering
        clusters = points_clustering(points_array, eps=0.5, min_samples=3)
        
        # Create polygons for each cluster
        polygons = []
        for cluster_points in clusters.values():
            # Create concave hull for each cluster
            cluster_poly = concave_hull(Polygon(cluster_points), 0.4)
            if cluster_poly.is_valid and not cluster_poly.is_empty:
                polygons.append(cluster_poly)
        
        # Create the final geometry
        if len(polygons) == 0:
            print(f"Warning: No valid polygons created for region {region_idx}")
            continue
        elif len(polygons) == 1:
            # Single polygon
            final_geometry = polygons[0]
        else:
            # Multiple polygons - create MultiPolygon
            final_geometry = MultiPolygon(polygons)
        
        # Store the WKT representation
        gfed_wkt_map[region_idx] = shapely.wkt.dumps(final_geometry)

    # Add the GFED regions to the dataframe
    for region_idx, region_name in gfed_areas:
        region_wkt = gfed_wkt_map.get(region_idx)
        # Use the region index as the name and ISO code
        wkt_df.loc[len(wkt_df)] = [region_name, region_name, region_name, region_wkt]

    # Save the extracted columns to a new CSV file
    output_filename = f"{csv_dir}/wkt.csv"
    wkt_df.to_csv(output_filename, index=False)

if __name__ == "__main__":
    format_countries_wkt_csv_file()
