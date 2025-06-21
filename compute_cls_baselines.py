#!/usr/bin/env python3

import logging
import xarray as xr
import numpy as np
from tqdm import tqdm
import argparse

logger = logging.getLogger(__name__)


def main(args):
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)

    logger.info("Opening local cube zarr file: {}".format(args.cube_path))
    cube = xr.open_zarr(args.cube_path, consolidated=False)

    octodays = list(range(1, 47)) * (2021 - 2001 + 1)
    da_octodays = xr.DataArray(octodays, dims="time", coords={"time": cube["time"]})
    cube["octodays"] = da_octodays
    training_years = slice("2002-01-01", "2018-01-01")
    variable = "gwis_ba"

    # Create binary fire occurrence variable (1 if fire, 0 if no fire)
    fire_occurrence = (cube[variable] > 0.0).astype(int)
    
    logger.info("Computing baseline forecasts...")
    
    # Initialize output arrays with the same shape as the input data
    baseline_any_fire = xr.DataArray(
        np.zeros(shape=cube[variable].shape),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": cube["time"],
            "latitude": cube["latitude"],
            "longitude": cube["longitude"],
        },
        name="baseline_any_fire"
    )
    
    baseline_majority = xr.DataArray(
        np.zeros(shape=cube[variable].shape),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": cube["time"],
            "latitude": cube["latitude"],
            "longitude": cube["longitude"],
        },
        name="baseline_majority"
    )
    
    # Compute baselines using only training data (2002-2018)
    training_data = cube.sel(time=training_years)
    training_fire_occurrence = fire_occurrence.sel(time=training_years)

    # Pre-compute baselines for each octoday using training data
    logger.info("Pre-computing baselines for each octoday using training data...")
    baseline_any_fire_by_octoday = {}
    baseline_majority_by_octoday = {}

    for octoday in tqdm(range(1, 47), desc="Computing baselines by octoday"):
        octoday_mask = training_data["octodays"] == octoday
        if octoday_mask.sum() > 0:
            octoday_fire_data = training_fire_occurrence.where(octoday_mask, drop=True)
            
            # Baseline 1: Any fire - predict fire if there was fire in any training year
            any_fire_prediction = (octoday_fire_data.sum("time") > 0).astype(int)
            
            # Baseline 2: Majority rule - predict fire if majority of training years had fire
            total_years = octoday_mask.sum()
            fire_years = octoday_fire_data.sum("time")
            majority_prediction = (fire_years > (total_years / 2)).astype(int)
            
            baseline_any_fire_by_octoday[octoday] = any_fire_prediction
            baseline_majority_by_octoday[octoday] = majority_prediction
        else:
            # If no data for this octoday, predict no fire
            baseline_any_fire_by_octoday[octoday] = xr.zeros_like(fire_occurrence.isel(time=0))
            baseline_majority_by_octoday[octoday] = xr.zeros_like(fire_occurrence.isel(time=0))

    # Apply baselines to all time steps
    for i, time in tqdm(enumerate(cube["time"]), total=len(cube["time"]), 
                       desc="Applying baselines to all time steps"):
        current_octoday = (i % 46) + 1
        
        # Assign pre-computed predictions to current time step
        baseline_any_fire.loc[dict(time=time)] = baseline_any_fire_by_octoday[current_octoday]
        baseline_majority.loc[dict(time=time)] = baseline_majority_by_octoday[current_octoday]

    # Create output dataset
    output = xr.Dataset({
        "baseline_any_fire": baseline_any_fire,
        "baseline_majority": baseline_majority,
    })
    
    # Add attributes for documentation
    output["baseline_any_fire"].attrs = {
        "long_name": "Baseline forecast - any fire in previous years",
        "description": "Predicts fire if there was fire in the same 8-day period in any previous year",
    }
    
    output["baseline_majority"].attrs = {
        "long_name": "Baseline forecast - majority rule",
        "description": "Predicts fire if majority of previous years had fire in the same 8-day period",
    }
    
    logger.info("Saving output to: {}".format(args.output_path))
    output.to_zarr(args.output_path, mode="w")
    logger.info("Baseline forecasting completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fire Baseline Forecasting")
    parser.add_argument(
        "--cube-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="cube_path",
        default="cube.zarr",
        help="Input cube path (zarr format)",
    )
    parser.add_argument(
        "--output-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="output_path",
        default="cls_baselines.zarr",
        help="Output path for baseline forecasts (zarr format)",
    )
    parser.add_argument("--debug", dest="debug", action="store_true", 
                       help="Enable debug logging")
    parser.add_argument("--no-debug", dest="debug", action="store_false",
                       help="Disable debug logging")
    parser.set_defaults(debug=False)
    
    args = parser.parse_args()
    main(args)