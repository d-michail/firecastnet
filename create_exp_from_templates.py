#!/usr/bin/env python3
import os
import sys

HORIZONS = [1, 2, 4, 8, 16, 24]
PROJECT_ROOT = os.path.abspath(os.getcwd())  # Root directory of the project

def update_script_content(filepath, region, horizon):
    """Reads a script and replaces job-name and SHIFT lines."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.startswith("#SBATCH --job-name="):
            new_lines.append(f"#SBATCH --job-name={region}-h{horizon}\n")
        elif line.startswith("SHIFT="):
            new_lines.append(f'SHIFT="{horizon}"\n')
        else:
            new_lines.append(line)

    return new_lines

def create_experiment_dirs(region_name):
    base_dir = os.path.join("exp", region_name)
    os.makedirs(base_dir, exist_ok=True)

    template_job = os.path.join(PROJECT_ROOT, "job.sh")
    template_test = os.path.join(PROJECT_ROOT, "test.sh")

    if not os.path.isfile(template_job) or not os.path.isfile(template_test):
        raise FileNotFoundError("job.sh and/or test.sh not found in project root.")

    for horizon in HORIZONS:
        horizon_dir = os.path.join(base_dir, f"h{horizon}")
        os.makedirs(horizon_dir, exist_ok=True)

        for template_file in [template_job, template_test]:
            updated_content = update_script_content(template_file, region_name, horizon)
            dest_file = os.path.join(horizon_dir, os.path.basename(template_file))
            with open(dest_file, "w") as f:
                f.writelines(updated_content)
            os.chmod(dest_file, 0o755)  # Make executable

        print(f"Created {horizon_dir} with job.sh and test.sh")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <region_name>")
        sys.exit(1)

    region = sys.argv[1]
    create_experiment_dirs(region)
