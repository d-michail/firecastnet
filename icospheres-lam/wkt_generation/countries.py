"""
Country and Continent Processing

This module contains functions for processing country and continent geometries
from Natural Earth datasets and generating WKT representations.
"""

import pandas as pd
import shapely
import shapely.wkt
from shapely.geometry import Polygon, MultiPolygon
from shapely import concave_hull

from .geometry_utils import reshape_wkt_coords


def process_countries(csv_dir="icospheres-lam/csv"):
    """
    Process country geometries from Natural Earth dataset.
    
    This function loads country WKT geometries from a CSV file and transforms
    the coordinates from (lon, lat) to (lat, lon) format.
    
    Args:
        csv_dir (str): Directory containing the CSV files
        
    Returns:
        pandas.DataFrame: DataFrame with country data including transformed WKT geometries
    """
    print("Copying wkt geometries of countries...")
    countries_csv = f"{csv_dir}/110m-admin-0-countries.csv"
    
    df = pd.read_csv(countries_csv, encoding='latin1', sep=';', quotechar='"')

    # Define the columns you want to extract
    columns_to_extract = ["SUBUNIT", "SU_A3", "NAME", 'WKT']
    df = df[columns_to_extract]
    
    # Transform WKT geometries
    wkt = df["WKT"]
    wkt = wkt.apply(lambda x: shapely.wkt.loads(x))
    wkt = wkt.apply(lambda x: reshape_wkt_coords(x))
    wkt = wkt.apply(lambda x: shapely.wkt.dumps(x))
    df["WKT"] = wkt
    
    return df.copy()


def process_continents(wkt_df, csv_dir="icospheres-lam/csv"):
    """
    Process continent geometries by aggregating country geometries.
    
    This function groups countries by continent and creates concave hulls
    for each continent using the constituent country geometries.
    
    Args:
        wkt_df (pandas.DataFrame): DataFrame containing country WKT data
        csv_dir (str): Directory containing the CSV files
        
    Returns:
        list: List of dictionaries containing continent data with WKT geometries
    """
    print("\nProcessing continents...")
    continents_csv = f"{csv_dir}/continent-country.csv"
    
    df = pd.read_csv(continents_csv, encoding='utf-8', sep=',', quotechar='"', keep_default_na=False)
    
    continent_isos = df['C-ISO3'].unique().tolist()
    continents_dict = {}
    
    # Process each continent
    for iso in continent_isos:
        continents_dict[iso] = {}
        countries = df[df['C-ISO3'] == iso]
        countries_wkt = wkt_df[wkt_df['SU_A3'].isin(countries["ISO3"])]['WKT'].apply(
            lambda x: shapely.wkt.loads(x)
        ).to_list()
        continent_name = countries.values[0][2]
        continent_wkts = []
        
        # Extract individual polygons from countries
        for wkt in countries_wkt:
            if isinstance(wkt, Polygon):
                continent_wkts.append(wkt)
            elif isinstance(wkt, MultiPolygon):
                continent_wkts.extend(wkt.geoms)

        # Initialize the continent dictionary
        continents_dict[iso] = {"wkt": continent_wkts, "name": continent_name, "iso": iso}
    
    # Create concave hulls for continents
    continent_data = []
    for iso in continents_dict:
        continent = continents_dict[iso]
        
        # Concatenate all the polygons into a single MultiPolygon and create a concave hull
        continent_hull = concave_hull(MultiPolygon(continent["wkt"]), 0.4)
        continent_wkt = shapely.wkt.dumps(continent_hull)
        
        continent_data.append({
            "SUBUNIT": continent["name"],
            "SU_A3": continent["iso"],
            "NAME": continent["name"],
            "WKT": continent_wkt
        })

    return continent_data