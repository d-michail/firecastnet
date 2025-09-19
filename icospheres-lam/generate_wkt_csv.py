import argparse
import pandas as pd
import os

# Import functions from wkt_generation package
from wkt_generation.countries import process_countries, process_continents
from wkt_generation.gfed import process_gfed_regions
from wkt_generation.shapefiles import process_shapefiles

csv_dir = './csv'


def format_countries_wkt_csv_file():
    """
    Generates a comprehensive CSV file containing WKT geometries for countries, continents, GFED regions, and custom shapefiles.

    This function processes multiple geographic data sources to create a unified CSV file with
    Well-Known Text (WKT) representations of geographic boundaries. The function always rewrites 
    the entire CSV file on each execution.
    
    Main operations:
    1. Extracts and transforms country geometries from a Natural Earth dataset
    2. Aggregates country geometries into continent-level polygons using concave hulls
    3. Processes GFED (Global Fire Emissions Database) regions from NetCDF data, using DBSCAN
       clustering to handle disconnected regions (such as island nations) as MultiPolygons
    4. Processes custom shapefiles from local files or URLs
    
    The function transforms all coordinates from (lon, lat) to (lat, lon) format for consistency
    and applies coordinate flipping where necessary to maintain proper geographic projections.
    
    Data Sources:
        - Countries: 110m-admin-0-countries.csv (Natural Earth dataset)
        - Continents: continent-country.csv (continent-country mapping)
        - GFED Regions: cube.zarr (NetCDF dataset with global fire emission regions)
        - Shapefiles: Custom shapefiles from local paths or URLs
    
    Output:
        Creates 'wkt.csv' with columns:
        - SUBUNIT: Administrative subunit name
        - SU_A3: ISO 3-letter country/region code
        - NAME: Display name
        - WKT: Well-Known Text geometry representation
    
    Command-line Options:
        --no-dbscan: Skip DBSCAN clustering for GFED regions
        --keep-temp: Keep temporary downloaded files
    
    Raises:
        FileNotFoundError: If required input files are not found
        ValueError: If geometries cannot be parsed or are invalid
        
    Note:
        The function prints progress information during processing and provides
        detailed feedback about what data types are being processed.
    """
    # Ensure CSV directory exists
    os.makedirs(csv_dir, exist_ok=True)
    
    # Prepare all new data
    all_new_data = []
        
    # --------- WKT BY GFED REGIONS ---------
    print("Processing GFED regions...")
    use_clustering = not args.no_dbscan
    gfed_data = process_gfed_regions(use_clustering=use_clustering)
    all_new_data.extend(gfed_data)
    
    # --------- WKT BY SHAPEFILES ---------
    try:                
        shapefile_wkt = process_shapefiles()
        all_new_data.extend(shapefile_wkt)
    except Exception as e:
        print(f"Error processing shapefiles: {e}")

    # Create new CSV file (always rewrite)
    output_filename = f"{csv_dir}/wkt.csv"
    print(f"Creating CSV file: {output_filename}")
    final_df = pd.DataFrame(all_new_data)
    
    # Save the final dataframe
    final_df.to_csv(output_filename, index=False)
    
    print(f"\nWKT CSV file saved successfully: {output_filename}")
    print(f"Total regions in file: {len(final_df)}")

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(
        description="Generate WKT CSV file for countries, continents, GFED regions, and custom shapefiles.",
        epilog="""
Examples:
  # Process all data types
  python generate_wkt_csv.py
 
  # Process without DBSCAN clustering for GFED regions
  python generate_wkt_csv.py --no-dbscan
          """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    args_parser.add_argument(
        "--no-dbscan",
        action="store_true",
        default=False,
        dest="no_dbscan",
        help="Skip DBSCAN clustering for GFED regions and use concave hulls directly."
    )
    args = args_parser.parse_args()
    format_countries_wkt_csv_file()
