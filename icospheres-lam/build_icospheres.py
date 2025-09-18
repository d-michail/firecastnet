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


if __name__ == "__main__":
    # Load the argument parser
    args = argparse_setup()

    if running_in_docker():
        from icosphere_generation.utils import load_yaml
        from icosphere_generation.process import execute_icosphere_generation
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
            config = configs[i]
            if "output" not in config:
                config["output"] = {}
            if "sphere" not in config:
                config["sphere"] = {}
            if args["outdir"]:
                config["output"]["directory"] = args["outdir"]

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
                save_icosphere(config, intersecting_mesh_layers_dict, filename + "_split_layers")
    else:
        import sys
        # Convert program arguments to string command
        raw_args = ' '.join(sys.argv[1:])
        run_command(f"sudo docker run --rm -v `pwd`:/models custom-pymesh:py3.7 /bin/bash -c 'cd /models && python build_icospheres.py {raw_args} && exit'")