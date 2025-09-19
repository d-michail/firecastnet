# WKT Generation Package

A modular Python package for generating Well-Known Text (WKT) geometries from various geographic data sources including countries, continents, and GFED regions.

## Overview

This package abstracts the functionality previously contained in a single `generate_wkt_csv.py` script into organized, reusable modules. It processes multiple geographic data sources to create unified CSV files with WKT representations of geographic boundaries.

## Package Structure

```
wkt_generation/
├── __init__.py          # Package initialization and main imports
├── geometry_utils.py    # Coordinate transformation and geometry utilities
├── clustering.py        # Point clustering algorithms (DBSCAN)
├── data_utils.py        # Data loading and processing utilities
├── countries.py         # Country and continent geometry processing
└── gfed.py             # GFED region geometry processing
```

## Modules

### geometry_utils.py
Contains utilities for coordinate transformation and geometry manipulation:
- `flip_coords(x, y)`: Flip coordinates from (x, y) to (y, x) format
- `reshape_wkt_coords(wkt_shape)`: Transform geometries from (lon, lat) to (lat, lon) format

### clustering.py
Provides clustering algorithms for geographic data:
- `points_clustering(points, eps=0.5, min_samples=3)`: DBSCAN clustering for separating disconnected regions

### data_utils.py
Data loading and processing utilities:
- `get_country(country_name, filename)`: Retrieve country information from CSV files

### countries.py
Functions for processing country and continent geometries:
- `process_countries(csv_dir)`: Process country geometries from Natural Earth dataset
- `process_continents(wkt_df, csv_dir)`: Create continent geometries by aggregating countries

### shapefiles.py
Functions for processing shapefiles:
- `extract_shapefile_features(shapefile_path, region_name, region_code)`: Convert shapefile features to WKT
- `process_shapefile_source(source, region_name, region_code)`: Process from local files or URLs (supports fiona's zip+http://)
- `process_multiple_shapefiles(sources)`: Batch processing of multiple shapefiles

## Usage

### Basic Usage

```python
from wkt_generation.countries import process_countries, process_continents
from wkt_generation.gfed import process_gfed_regions

# Process countries
countries_df = process_countries("path/to/csv/dir")

# Process continents
continents_data = process_continents(countries_df, "path/to/csv/dir")

# Process GFED regions
gfed_data = process_gfed_regions(use_clustering=True)
```

### Using Individual Utilities

```python
from wkt_generation import process_shapefile_source

# Local shapefile
local_data = process_shapefile_source("path/to/shapefile.shp", "RegionName", "REG")

# Remote shapefile (direct .shp file)
remote_data = process_shapefile_source("https://example.com/data.shp", "RemoteRegion", "REM")

# Remote zip archive (fiona handles automatically)
zip_data = process_shapefile_source("https://example.com/data.zip", "ZipRegion", "ZIP")

# Or explicitly specify zip protocol
zip_data = process_shapefile_source("zip+https://example.com/data.zip", "ZipRegion", "ZIP")
```

## Data Sources

The package processes data from:

1. **Countries**: 110m-admin-0-countries.csv (Natural Earth dataset)
2. **Continents**: continent-country.csv (continent-country mapping)
3. **GFED Regions**: cube.zarr (NetCDF dataset with global fire emission regions)

## Dependencies

- pandas
- numpy
- shapely
- xarray
- scikit-learn (for DBSCAN clustering)

## Migration from Original Script

The original `generate_wkt_csv.py` has been refactored to use this modular structure:

- Function definitions moved to appropriate modules
- Main processing logic simplified and modularized
- Imports updated to use the new package structure
- Better separation of concerns and code organization

## Output

The package generates CSV files with the following columns:
- `SUBUNIT`: Administrative subunit name
- `SU_A3`: ISO 3-letter country/region code  
- `NAME`: Display name
- `WKT`: Well-Known Text geometry representation

## Features

- **Coordinate Transformation**: Converts between (lon, lat) and (lat, lon) formats
- **Clustering Support**: Uses DBSCAN to handle disconnected regions (islands, archipelagos)
- **Multiple Data Sources**: Processes countries, continents, and GFED regions
- **Concave Hull Generation**: Creates realistic boundary shapes
- **Error Handling**: Robust processing with informative error messages