import os

def argparse_setup():
    import argparse
    parser = argparse.ArgumentParser(description="Generate an icosphere with adaptive mesh refinement.")
    parser.add_argument(
        "--config",
        default="./configs/config.yaml",
        dest="config_path",
        help="Path to the configuration file."
    )
    parser.add_argument(
        "--configs_all", 
        action="store_true",
        default=False,
        help="If set, process all config files in the ./configs/ directory."
    )
    parser.add_argument(
        "--outdir",
        default="./icospheres/",
        dest="outdir",
        help="Output directory for the generated icosphere files."
    )
    return vars(parser.parse_args())

def running_in_docker():
    return os.path.exists("/.dockerenv")

def run_command(command: str) -> str:
    import subprocess
    result = subprocess.run(
        command,
        shell=True,          # Use shell so string commands work
        text=True,           # Return output as str instead of bytes
        capture_output=True  # Capture stdout & stderr
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}:\n{result.stderr}")
    print(result.stdout)
    return result.stdout.strip()

def save_icosphere(config: dict, mesh, filename: str):
    import json

    directory = config["output"].get("directory", "./icospheres/")
    os.makedirs(directory, exist_ok=True, mode=0o755)
    location = directory + f"{filename}.json"
    with open(location, 'w') as f:
        json.dump(mesh, f)
    print(f"Generated icosphere saved to {location}")

    if config and config.get("gzip", False):
        from icosphere_generation.utils import gzip_file
        gzip_file(location)
        print(f"Generated icosphere saved to {location}.gz")
        
def initialize_PolygonStructures(config: dict):
    import pandas as pd
    from icosphere_generation.PolygonStructure import PolygonStructure
    from shapely.wkt import loads as wkt_loads
    from typing import List
    
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

def execute_icosphere_generation(config: dict):
    from icosphere_generation.preprocess import generate_initial_mesh, polygon_structures_preprocess
    from icosphere_generation.utils import generate_icosphere_file_code
    from icosphere_generation.mesh_operations import generate_icosphere

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
        intersection_layers=config.get("intersection_layers", False),
            radius=radius,
            center=center
        )
    
    return icospheres_dict, mesh_layers_dict, intersecting_mesh_layers_dict, file_code


if __name__ == "__main__":
    # Load the argument parser
    args = argparse_setup()

    if running_in_docker():
        from icosphere_generation.utils import load_yaml
        if not os.path.exists("./configs/"):
            os.makedirs("./configs/", exist_ok=True)
        if not os.path.exists(args["config_path"]):
            raise FileNotFoundError(f"Config file {args['config_path']} does not exist.")

        # Determine which config files to process
        configs = []
        if args["configs_all"]:
            configs = [os.path.join("./configs/", f) for f in os.listdir("./configs/") if f.endswith(".yaml")]
        else:
            configs.append(args["config_path"])

        # Load and possibly modify each config
        for i, config in enumerate(configs):
            configs[i] = load_yaml(config)
            if args["outdir"]:
                config["output"]["directory"] = args["outdir"]
            if "output" not in config:
                config["output"] = {}
            if "sphere" not in config:
                config["sphere"] = {}

        for config in configs:
            # Execute the icosphere generation
            icospheres_dict, mesh_layers_dict, intersecting_mesh_layers_dict, file_code =\
                execute_icosphere_generation(config)
            
            # Save the icosphere to a file
            filename = config["output"].get("filename", file_code)
            save_icosphere(config, icospheres_dict, filename)

            # Save mesh layers if they were generated
            if mesh_layers_dict:
                save_icosphere(config, mesh_layers_dict, filename + "_layers")
            if intersecting_mesh_layers_dict:
                save_icosphere(config, intersecting_mesh_layers_dict, filename + "_intersection_layers")
    else:
        import sys
        # Convert program arguments to string command
        raw_args = ' '.join(sys.argv[1:])
        run_command(f"sudo docker run --rm -v `pwd`:/models custom-pymesh:py3.7 /bin/bash -c 'cd /models && python build_icospheres.py {raw_args} && exit'")