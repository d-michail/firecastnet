#!/usr/bin/env python3

import logging
import xarray as xr
import numpy as np
from tqdm import tqdm
import json
import torch
from tqdm import tqdm
import xarray as xr
from tqdm import tqdm
import numpy as np
import json
import argparse
from seasfire.firecastnet_lit import FireCastNetLit

logger = logging.getLogger(__name__)


def main(args):
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)

    # load mean and std
    mean_std_dict_filename = f"cube_mean_std_dict_{args.target_shift}.json"
    logger.info("Opening mean-std statistics = {}".format(mean_std_dict_filename))
    mean_std_dict = None
    with open(mean_std_dict_filename, "r") as f:
        mean_std_dict = json.load(f)

    input_vars = [
        "mslp",
        "tp",
        "vpd",
        "sst",
        "t2m_mean",
        "ssrd",
        "swvl1",
        "lst_day",
        "ndvi",
        "pop_dens",
    ]
    lsm_var = "lsm"
    static_vars = [lsm_var]
    log_preprocess_input_vars = ["tp", "pop_dens"]
    target_var = "gwis_ba"

    logger.info("Opening local cube zarr file: {}".format(args.cube_path))
    cube = xr.open_zarr(args.cube_path, consolidated=False)

    for var_name in log_preprocess_input_vars:
        logger.info("Log-transforming input var: {}".format(var_name))
        cube[var_name] = xr.DataArray(
            np.log(1.0 + cube[var_name].values),
            coords=cube[var_name].coords,
            dims=cube[var_name].dims,
            attrs=cube[var_name].attrs,
        )

    for static_v in static_vars:
        logger.info(
            "Expanding time dimension on static variable = {}.".format(static_v)
        )
        cube[static_v] = cube[static_v].expand_dims(dim={"time": cube.time}, axis=0)

    # normalize input variables
    for var in input_vars:
        var_mean = mean_std_dict[f"{var}_mean"]
        var_std = mean_std_dict[f"{var}_std"]
        cube[var] = (cube[var] - var_mean) / var_std

    # keep only needed vars
    ds = cube[input_vars + static_vars]
    ds = ds.fillna(-1)

    # shift time inputs forward in time
    logger.info(f"Shifting inputs by {args.target_shift}.")
    for var in input_vars:
        if args.target_shift > 0:
            ds[var] = ds[var].shift(time=args.target_shift, fill_value=0)

    # load model from checkpoint
    logger.info(f"Loading model from ckpt = {args.ckpt_path}")
    model = FireCastNetLit.load_from_checkpoint(args.ckpt_path)
    model.eval()
    model.dglTo(model.device)

    # prepare predictions for storage
    predictions = np.zeros_like(
        cube[target_var]
    )

    logger.info(f"Will create samples for [{args.start_time}, {args.end_time}]")
    ds_selected = ds.sel(time=slice(args.start_time, args.end_time))
    ds_selected_time_indexes = ds.get_index("time").get_indexer(ds_selected["time"])


    for t_index in tqdm(ds_selected_time_indexes, desc="Processing samples"):        
        if t_index < args.timeseries - 1:
            continue

        sample = ds.isel(time=slice(t_index - args.timeseries + 1, t_index + 1))
        sample_tensor = torch.tensor(sample.to_array().values, dtype=torch.float32).to(
            model.device
        )
        sample_tensor = sample_tensor.unsqueeze(0)

        with torch.no_grad():
            prediction = model.predict_step(sample_tensor)

        prediction = prediction.cpu().numpy()
        predictions[t_index] = prediction.squeeze()

        all_nan = np.isnan(prediction).all()
        if all_nan: 
            logger.warning("All prediction values are NaN")

    da = xr.DataArray(
        predictions,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": ds["time"],
            "latitude": ds["latitude"],
            "longitude": ds["longitude"],
        },
    )

    da = da.where(cube[lsm_var] > 0.1, np.nan)

    # Filter predictions by GFED region if specified
    if args.gfed_region is not None:
        logger.info(f"Filtering predictions for GFED region: {args.gfed_region}")
        if "gfed_region" in cube:
            # Map region name to integer code
            region_name_to_int = {
                "OCEAN": 0,
                "BONA": 1,
                "TENA": 2,
                "CEAM": 3,
                "NHSA": 4,
                "SHSA": 5,
                "EURO": 6,
                "MIDE": 7,
                "NHAF": 8,
                "SHAF": 9,
                "BOAS": 10,
                "CEAS": 11,
                "SEAS": 12,
                "EQAS": 13,
                "AUST": 14,
            }
            region_code = region_name_to_int.get(args.gfed_region)
            if region_code is not None:
                region_mask = (cube["gfed_region"].values == region_code)
                da = da.where(region_mask, np.nan)
            else:
                logger.warning(f"GFED region '{args.gfed_region}' not found in mapping. No filtering applied.")
        else:
            logger.warning("Cube does not contain 'gfed_region' variable. No filtering applied.")

    output_var_name = f"{args.output_var_prefix}_{args.target_shift}"
    
    logger.info(f"Creating new zarr store at {args.output_path}")
    ds_output = xr.Dataset({output_var_name: da})
    ds_output.to_zarr(args.output_path, mode="w")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inference FireCastNet")
    parser.add_argument(
        "--cube-path",
        metavar="KEY",
        type=str,
        action="store",
        dest="cube_path",
        default="cube.zarr",
        help="Cube path",
    )
    parser.add_argument(
        "--ckpt-path",
        metavar="KEY",
        type=str,
        action="store",
        dest="ckpt_path",
        default="best.ckpt",
        help="Checkpoint path",
    )
    parser.add_argument(
        "--target-shift",
        metavar="KEY",
        type=int,
        action="store",
        dest="target_shift",
        default=1,
        help="Target shift",
    )
    parser.add_argument(
        "--timeseries",
        metavar="KEY",
        type=int,
        action="store",
        dest="timeseries",
        default=24,
        help="Timeseries length",
    )
    parser.add_argument(
        "--start-time",
        metavar="KEY",
        type=str,
        action="store",
        dest="start_time",
        default="2019-01-01",
        help="Start time",
    )
    parser.add_argument(
        "--end-time",
        metavar="KEY",
        type=str,
        action="store",
        dest="end_time",
        default="2020-01-01",
        help="End time",
    )
    parser.add_argument(
        "--output-var-prefix",
        metavar="KEY",
        type=str,
        action="store",
        dest="output_var_prefix",
        default="predictions_cls_ba",
        help="Prediction variable prefix",
    )        
    parser.add_argument(
        "--output-path",
        metavar="KEY",
        type=str,
        action="store",
        dest="output_path",
        default="predictions.zarr",
        help="Output path",
    )
    parser.add_argument(
        "--gfed-region",
        metavar="KEY",
        type=str,
        action="store",
        dest="gfed_region",
        default=None,
        help="GFED region name to filter predictions (e.g., 'NHAF'). If not set, no filtering is applied.",
    )
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    args = parser.parse_args()

    main(args)
