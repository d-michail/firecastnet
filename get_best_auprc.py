#!/usr/bin/env python3
import os
import re
import sys

def find_best_ckpt(relative_dir):
    # Step 1: Go to directory from home
    home_dir = os.path.expanduser("~")
    target_dir = os.path.join(home_dir, "Projects", "Python", "firecastnet", "exp", relative_dir, "checkpoints")

    if not os.path.isdir(target_dir):
        raise NotADirectoryError(f"Directory not found: {target_dir}")

    # Step 2: Regex pattern to match ckpt files
    pattern = re.compile(r"epoch=(\d+)-val_auprc=([0-9.]+)\.ckpt$")

    best_file = None
    best_score = float("-inf")

    # Step 3: Iterate over files
    for filename in os.listdir(target_dir):
        match = pattern.match(filename)
        if match:
            auprc_value = float(match.group(2))
            if auprc_value > best_score:
                best_score = auprc_value
                best_file = filename

    if best_file is None:
        raise FileNotFoundError("No matching checkpoint files found.")

    return best_file

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <relative_directory_from_home>")
        sys.exit(1)

    try:
        best_ckpt = find_best_ckpt(sys.argv[1])
        print(best_ckpt)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
