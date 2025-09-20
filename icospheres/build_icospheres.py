#!/usr/bin/env python
import numpy as np

def running_in_docker():
    import os
    return os.path.exists("/.dockerenv")


def run_command(command: str) -> None:
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

def generate_and_save_icospheres(
    save_path: str = "icospheres_0_1_2_3_4_5_6.json", levels=[0, 1, 2, 3, 4, 5, 6]
) -> None:  # pragma: no cover
    try:
        import pymesh
    except ImportError:
        Warning("Pymesh is not installed. Please install it to generate icospheres.")
    import json

    """Generate icospheres from level 0 to 6 (inclusive) and save them to a json file.

    Parameters
    ----------
    path : str
        Path to save the json file.
    """
    radius = 1
    center = np.array((0, 0, 0))
    icospheres = {"vertices": [], "faces": []}

    # Generate icospheres from level 0 to 6 (inclusive)
    cur = 0
    for order in levels:
        icosphere = pymesh.generate_icosphere(radius, center, refinement_order=order)
        icospheres["order_" + str(cur) + "_vertices"] = icosphere.vertices
        icospheres["order_" + str(cur) + "_faces"] = icosphere.faces
        icosphere.add_attribute("face_centroid")
        icospheres["order_" + str(cur) + "_face_centroid"] = (
            icosphere.get_face_attribute("face_centroid")
        )
        cur += 1

    # save icosphere vertices and faces to a json file
    icospheres_dict = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in icospheres.items()
    }
    with open(save_path, "w") as f:
        json.dump(icospheres_dict, f)


if __name__ == "__main__":
    if running_in_docker():
        # Running inside Docker container - execute the actual generation
        generate_and_save_icospheres("icospheres_0_1.json", [0, 1])
        generate_and_save_icospheres("icospheres_0_1_2.json", [0, 1, 2])
        generate_and_save_icospheres("icospheres_0_1_2_3.json", [0, 1, 2, 3])
        generate_and_save_icospheres("icospheres_0_1_2_3_4.json", [0, 1, 2, 3, 4])
        generate_and_save_icospheres("icospheres_0_1_2_3_4_5.json", [0, 1, 2, 3, 4, 5])
        generate_and_save_icospheres("icospheres_0_1_2_3_4_5_6.json", [0, 1, 2, 3, 4, 5, 6])
        generate_and_save_icospheres("icospheres_1_3_5.json", [1, 3, 5])
        generate_and_save_icospheres("icospheres_2_3_4.json", [2, 3, 4])
        generate_and_save_icospheres("icospheres_2_3_4_5.json", [2, 3, 4, 5])
    else:
        # Running outside Docker - run the script inside Docker container
        import sys
        raw_args = ' '.join(sys.argv[1:])
        run_command(f"sudo docker run --rm -v `pwd`:/models custom-pymesh:py3.7 /bin/bash -c 'cd /models && python build_icospheres.py {raw_args} && exit'")
