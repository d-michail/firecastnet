# Build icospheres

## Setup

```Bash
# Generate WKT csv file of each country, continent and GFED region
python generate_wkt_csv.py --cube_path /path/to/seasfire_cube

# Or to specify the output directory
python generate_wkt_csv.py --csv_dir /path/to/output --cube_path /path/to/seasfire_cube

# Build the Docker image
docker build -t custom-pymesh:py3.7 -f Dockerfile .
```

## Build the icosphere with from a config file

```Bash
# Build the icosphere using the default config file
python ./build_icospheres.py

# Or specify the config file and/or output directory
python ./build_icospheres.py --config config.yaml

# Process all config files in the ./configs/ directory
python ./build_icospheres.py --configs_all 

# Override the output directory for all generated files
python ./build_icospheres.py --configs_all --outdir ./my_output_directory/
```

## Visualize the generated icosphere file

```Bash
# Visualize the generated icosphere file
python ./icosphere_vis.py --config /path/to/config.yaml
```

## Configuration

To configure the icosphere build process, you can edit the `config.yaml` file the following structure:

```yaml
sphere:
    # The refinement order of the sphere (default is 3)
    refinement_order: 3 
    # The radius of the sphere (default is 1.0)
    radius: 1.0 
    # The center of the sphere (default is [0.0, 0.0, 0.0])
    center: [0.0, 0.0, 0.0] 
refinement_targets:
  # Alpha-3 codes from ISO 3166-1
  - target_code: "USA" 
    # The final refinement order of the target
    refinement_order: 5 
    # The type of refinement to apply (Default: "none")
    refiniment_type: "uniform | block | none" 
    # Refinemnt buffer amount - Only applies to uniform refinement type
    buffer_factor: 50.0
    # Refinement buffer unit of measure (Default: "km")
    buffer_unit: "km | percent"
    # Interest of the target (Default: true)
    interest: true

  # Administrative level 1 names with country code (e.g., "California, USA")
  - target_code: "California, USA | Ontario, CAN | Bavaria, DEU" 

  # Continent codes from ISO 3166-1
  - target_code: "AF | AN | AS | EU | NA | OC | SA" 
  
  # GFED region codes
  - target_code: "BONA | TENA | CEAM | NHSA | SHSA | EURO | MIDE | NHAF | SHAF | BOAS | CEAS | SEAS | EQAS | AUST"
  
  # Bring your own Polygon/MultiPolygon WKT
  - custom_wkt: "..." 
    ... # Same structure as above

gzip: true # Whether to gzip the output json file (default is false)
split_layers: true # Whether to generate mesh with each refinement layer
output:
    directory: "./icospheres/" # The output directory (default is ./icospheres/)
    filename: "icosphere" # The base filename for the output files (default is an auto-generated name based on configuration)
```
