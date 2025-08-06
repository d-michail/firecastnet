# Build icospheres

## Setup

```Bash
# From the root of the repository
# Generate WKT csv file of each country, continent and GFED region
python icospheres-lam/generate_wkt_csv.py

# Build the Docker image
docker build -t custom-pymesh:py3.7 -f Dockerfile .
```

## Build the icosphere with from a config file

```Bash
cd seasfire/icospheres-lam
docker run -it --rm -v `pwd`:/icospheres custom-pymesh:py3.7 /bin/bash
cd /icospheres
python ./build_icospheres.py
exit
```

To configure the icosphere build process, you can edit the `config.yaml` file the following structure:

```yaml
sphere:
    refinement_order: 3 # The refinement order of the sphere (default is 3)
    radius: 1.0 # The radius of the sphere (default is Φ phi = 1.6180339887)
    center: [0.0, 0.0, 0.0] # The center of the sphere (default is [0.0, 0.0, 0.0])
refinement_targets:
  # Alpha-3 codes from ISO 3166-1
  - country_code: "USA" 
    # The final refinement order of the target
    refinement_order: 5 
    # The type of refinement to apply (Default: "none")
    refiniment_type: "uniform | block | none" 
    # Refinemnt buffer amount - Only applies to uniform refinement type
    refinement_buffer: 50.0
    # Refinement buffer unit of measure (Default: "km")
    refinement_buffer_unit: "km | percent"
    # Interest of the target (Default: true)
    interest: true

  # Continent codes from ISO 3166-1
  - country_code: "AF | AN | AS | EU | NA | OC | SA" 
    ... # Same structure as above
  
  # GFED region codes
  - gfed_code: "BONA | TENA | CEAM | NHSA | SHSA | EURO | MIDE | NHAF | SHAF | BOAS | CEAS | SEAS | EQAS | AUST"
    ... # Same structure as above
  
  # Bring your own Polygon/MultiPolygon WKT
  - custom_wkt: "..." 
    ... # Same structure as above
gzip: true # Whether to gzip the output json file (default is false)

```
