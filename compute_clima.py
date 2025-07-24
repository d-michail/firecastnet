import logging
import xarray as xr
import numpy as np
from tqdm import tqdm
from tqdm import tqdm
import xarray as xr
from tqdm import tqdm
import numpy as np
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

    if args.log_target_var:
        cube[variable] = xr.DataArray(
            np.log(1.0 + cube[variable].values),
            coords=cube[variable].coords,
            dims=cube[variable].dims,
            attrs=cube[variable].attrs,
        )    

    # calculate climatology mean by octon for years 2002-2018
    clim_ba_octodays_mean = (
        cube.sel(time=training_years).groupby("octodays").mean("time")[variable]
    )
    clim_ba_octodays_std = (
        cube.sel(time=training_years).groupby("octodays").std("time")[variable]
    )

    cube[f"clim_{variable}_octodays_mean"] = xr.DataArray(
        np.zeros(shape=cube["gwis_ba"].shape),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": cube["time"],
            "latitude": cube["latitude"],
            "longitude": cube["longitude"],
        },
    )
    cube[f"clim_{variable}_octodays_std"] = xr.DataArray(
        np.zeros(shape=cube["gwis_ba"].shape),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": cube["time"],
            "latitude": cube["latitude"],
            "longitude": cube["longitude"],
        },
    )

    clim_ba_octodays_mean = clim_ba_octodays_mean.fillna(0).load()
    clim_ba_octodays_std = clim_ba_octodays_std.fillna(0).load()

    for i, time in tqdm(enumerate(cube["time"]), total=len(cube["time"])):
        octoday_idx = i % 46
        cube[f"clim_{variable}_octodays_mean"].loc[dict(time=time)] = (
            clim_ba_octodays_mean.sel(octodays=1 + octoday_idx)
        )
        cube[f"clim_{variable}_octodays_std"].loc[dict(time=time)] = (
            clim_ba_octodays_std.sel(octodays=1 + octoday_idx)
        )

    output = xr.Dataset(
        {
            f"clim_{variable}_octodays_mean": cube[f"clim_{variable}_octodays_mean"],
            f"clim_{variable}_octodays_std": cube[f"clim_{variable}_octodays_std"],
        }
    )
    output.to_zarr(args.output_path, mode="w")

    # ---------------------------------------------------------------------
    # === NEW BLOCK: fire occurrence prediction baselines ================
    # ---------------------------------------------------------------------
    fire_bool = cube[variable] > 0
    fire_int = fire_bool.astype("int8").assign_coords(octodays=cube["octodays"])

    prev_fire_count = fire_int.groupby("octodays").cumsum("time") - fire_int
    prev_total_years = xr.ones_like(fire_int).groupby("octodays").cumsum("time") - 1

    pred_fire_any = (prev_fire_count > 0).astype("uint8")
    pred_fire_majority = (
        prev_fire_count > (prev_total_years - prev_fire_count)
    ).astype("uint8")

    # Append these two to disk as a new Zarr dataset using same path
    fire_baselines = xr.Dataset(
        {
            "pred_fire_any": pred_fire_any,
            "pred_fire_majority": pred_fire_majority,
        }
    )
    fire_baselines.to_zarr(args.output_path, mode="a")
    # ---------------------------------------------------------------------


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Climatology")
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
        "--output-path",
        metavar="KEY",
        type=str,
        action="store",
        dest="output_path",
        default="climatology.zarr",
        help="Output path",
    )
    parser.add_argument(
        "--log-target-var",
        dest="log_target_var",
        action="store_true",
    )
    parser.add_argument(
        "--no-log-target-var",
        dest="log_target_var",
        action="store_false",
    )
    parser.set_defaults(log_target_var=False)    
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    args = parser.parse_args()

    main(args)
