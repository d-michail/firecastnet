#!/usr/bin/env python3

import logging
import xarray as xr
import torch
from torchmetrics.classification import AveragePrecision, Accuracy, F1Score
from tqdm import tqdm
import argparse

logger = logging.getLogger(__name__)

# GFED region codes
REGION_NAME_TO_INT = {
    "OCEAN": 0, "BONA": 1, "TENA": 2, "CEAM": 3, "NHSA": 4,
    "SHSA": 5, "EURO": 6, "MIDE": 7, "NHAF": 8, "SHAF": 9,
    "BOAS": 10, "CEAS": 11, "SEAS": 12, "EQAS": 13, "AUST": 14
}

def main(args): 
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)

    logger.info(f"Loading target cube from: {args.cube_path}")
    cube = xr.open_zarr(args.cube_path, consolidated=False)
    logger.info(f"Loading prediction cube from: {args.pred_path}")
    preds = xr.open_zarr(args.pred_path, consolidated=False)
    predictions = preds[args.pred_var]

    gfed_regions = cube["gfed_region"]
    lsm = cube["lsm"]
    target = cube[args.target_var]

    times = cube["time"].sel(time=slice(args.start_time, args.end_time))

    # Store metrics
    metrics_results = {}

    # Create a list of regions to compute metrics for
    regions_to_compute = []
    if args.gfed_region:
        regions_to_compute.append(args.gfed_region)
    else:
        regions_to_compute.append("GLOBAL")
        regions_to_compute.extend(REGION_NAME_TO_INT.keys())

    # Compute metrics for each region in the list
    for region_name in regions_to_compute:
        if region_name == "GLOBAL":
            global_metrics = {
                "accuracy": Accuracy(task="binary"),
                "f1": F1Score(task="binary"),
                "auprc": AveragePrecision(task="binary")
            }

            lsm_mask = torch.tensor((lsm.values > 0.1))
            for time in tqdm(times.values, desc="Evaluating global metrics"):
                t_pred = torch.tensor(predictions.sel(time=time).where(lsm_mask).fillna(0).values)
                t_target = torch.tensor(target.sel(time=time).where(lsm_mask).fillna(0).values)
                t_target = torch.where(t_target != 0, 1, 0)

                for metric in global_metrics.values():
                    metric.update(t_pred.flatten(), t_target.flatten())

            metrics_results["global"] = {
                name: metric.compute().item() for name, metric in global_metrics.items()
            }
        else:
            region_code = REGION_NAME_TO_INT[region_name]
            region_metrics = {
                "accuracy": Accuracy(task="binary"),
                "f1": F1Score(task="binary"),
                "auprc": AveragePrecision(task="binary")
            }

            # Apply all masks (GFED + LSM)
            region_mask = gfed_regions == region_code
            mask = region_mask & (lsm > 0.1)

            for time in tqdm(times.values, desc=f"Evaluating {region_name}"):
                t_pred = torch.tensor(predictions.sel(time=time).where(mask).fillna(0).values)
                t_target = torch.tensor(target.sel(time=time).where(mask).fillna(0).values)
                t_target = torch.where(t_target != 0, 1, 0)

                for metric in region_metrics.values():
                    metric.update(t_pred.flatten(), t_target.flatten())

            metrics_results[region_name] = {
                name: metric.compute().item() for name, metric in region_metrics.items()
            }

    # Output all metrics at the end with header check
    if "global" in metrics_results:
        logger.info("\n=== Global Metrics ===")
        for name, value in metrics_results["global"].items():
            logger.info(f"{name.upper()}: {value:.4f}")

    if any(region_name != "global" for region_name in metrics_results):
        logger.info("\n=== Per-Region Metrics ===")
        for region_name, region_metrics in metrics_results.items():
            if region_name == "global":
                continue
            logger.info(f"Region: {region_name}")
            for name, value in region_metrics.items():
                logger.info(f"  {name.upper()}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics globally and per GFED region.")
    parser.add_argument("--cube-path", type=str, required=True, help="Path to the cube with targets and GFED region mask.")
    parser.add_argument("--pred-path", type=str, required=True, help="Path to the cube with predictions.")
    parser.add_argument("--target-var", type=str, default="gwis_ba", help="Name of the target variable.")
    parser.add_argument("--pred-var", type=str, default="predictions_cls_ba", help="Name of the predictions variable.")
    parser.add_argument("--start-time", type=str, default="2019-01-01", help="Start time for evaluation.")
    parser.add_argument("--end-time", type=str, default="2020-01-01", help="End time for evaluation.")
    parser.add_argument("--gfed-region", type=str, default=None, choices=list(REGION_NAME_TO_INT.keys()), help="Specify a single GFED region to compute metrics for.")
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no-debug", dest="debug", action="store_false")    
    args = parser.parse_args()

    main(args)
