import argparse
import pandas as pd
import os

from wkt_generation.gfed import process_gfed_regions
from wkt_generation.shapefiles import process_shapefiles

def format_countries_wkt_csv_file():
    # Ensure CSV directory exists
    os.makedirs(args.csv_dir, exist_ok=True)
    
    # Prepare all new data
    all_new_data = []
        
    # --------- WKT BY GFED REGIONS ---------
    use_clustering = not args.no_dbscan
    gfed_data = process_gfed_regions(args.cube_path, use_clustering=use_clustering)
    all_new_data.extend(gfed_data)
    
    # --------- WKT BY SHAPEFILES ---------
    try:                
        shapefile_wkt = process_shapefiles()
        all_new_data.extend(shapefile_wkt)
    except Exception as e:
        print(f"Error processing shapefiles: {e}")

    # Create new CSV file (always rewrite)
    output_filename = f"{args.csv_dir}/wkt.csv"
    print(f"Creating CSV file: {output_filename}")
    final_df = pd.DataFrame(all_new_data)
    
    # Save the final dataframe
    final_df.to_csv(output_filename, index=False)
    
    print(f"\nWKT CSV file saved successfully: {output_filename}")
    print(f"Total regions in file: {len(final_df)}")

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(
        description="Generate WKT CSV file for different geographic regions.",
        epilog="""
Examples:
  python generate_wkt_csv.py --cube_path /path/to/cube
  python generate_wkt_csv.py --cube_path /path/to/cube --no-dbscan
  python generate_wkt_csv.py --cube_path /path/to/cube --csv_dir /path/to/output
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    args_parser.add_argument(
        "--no-dbscan",
        action="store_true",
        help="Skip DBSCAN clustering for GFED regions."
    )
    args_parser.add_argument(
        "--cube_path",
        type=str,
        required=True,
        help="Required path to the SeasFire cube directory."
    )
    args_parser.add_argument(
        "--csv_dir",
        type=str,
        default="./csv",
        help="Output directory for CSV file."
    )
    args = args_parser.parse_args()
    format_countries_wkt_csv_file()
