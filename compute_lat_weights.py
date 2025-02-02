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


# the code is taken from https://github.com/google-deepmind/graphcast/blob/main/graphcast/losses.py

# to get the weights on the seasfire dataset you use the following line


def normalized_latitude_weights(data: xr.DataArray) -> xr.DataArray:
    """Weights based on latitude, roughly proportional to grid cell area.
    This method supports two use cases only (both for equispaced values):
    * Latitude values such that the closest value to the pole is at latitude
      (90 - d_lat/2), where d_lat is the difference between contiguous latitudes.
      For example: [-89, -87, -85, ..., 85, 87, 89]) (d_lat = 2)
      In this case each point with `lat` value represents a sphere slice between
      `lat - d_lat/2` and `lat + d_lat/2`, and the area of this slice would be
      proportional to:
      `sin(lat + d_lat/2) - sin(lat - d_lat/2) = 2 * sin(d_lat/2) * cos(lat)`, and
      we can simply omit the term `2 * sin(d_lat/2)` which is just a constant
      that cancels during normalization.
    * Latitude values that fall exactly at the poles.
      For example: [-90, -88, -86, ..., 86, 88, 90]) (d_lat = 2)
      In this case each point with `lat` value also represents
      a sphere slice between `lat - d_lat/2` and `lat + d_lat/2`,
      except for the points at the poles, that represent a slice between
      `90 - d_lat/2` and `90` or, `-90` and  `-90 + d_lat/2`.
      The areas of the first type of point are still proportional to:
      * sin(lat + d_lat/2) - sin(lat - d_lat/2) = 2 * sin(d_lat/2) * cos(lat)
      but for the points at the poles now is:
      * sin(90) - sin(90 - d_lat/2) = 2 * sin(d_lat/4) ^ 2
      and we will be using these weights, depending on whether we are looking at
      pole cells, or non-pole cells (omitting the common factor of 2 which will be
      absorbed by the normalization).
      It can be shown via a limit, or simple geometry, that in the small angles
      regime, the proportion of area per pole-point is equal to 1/8th
      the proportion of area covered by each of the nearest non-pole point, and we
      test for this in the test.
    Args:
      data: `DataArray` with latitude coordinates.
    Returns:
      Unit mean latitude weights.
    """
    latitude = data.coords['latitude']

    if np.any(np.isclose(np.abs(latitude), 90.)):
        weights = _weight_for_latitude_vector_with_poles(latitude)
    else:
        weights = _weight_for_latitude_vector_without_poles(latitude)

    return weights / weights.mean(skipna=False)


def _weight_for_latitude_vector_without_poles(latitude):
    """Weights for uniform latitudes of the form [+-90-+d/2, ..., -+90+-d/2]."""
    delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
    if (not np.isclose(np.max(latitude), 90 - delta_latitude / 2) or
            not np.isclose(np.min(latitude), -90 + delta_latitude / 2)):
        raise ValueError(
            f'Latitude vector {latitude} does not start/end at '
            '+- (90 - delta_latitude/2) degrees.')
    return np.cos(np.deg2rad(latitude))


def _weight_for_latitude_vector_with_poles(latitude):
    """Weights for uniform latitudes of the form [+- 90, ..., -+90]."""
    delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
    if (not np.isclose(np.max(latitude), 90.) or
            not np.isclose(np.min(latitude), -90.)):
        raise ValueError(
            f'Latitude vector {latitude} does not start/end at +- 90 degrees.')
    weights = np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(delta_latitude / 2))
    # The two checks above enough to guarantee that latitudes are sorted, so
    # the extremes are the poles
    weights[[0, -1]] = np.sin(np.deg2rad(delta_latitude / 4)) ** 2
    return weights


def _check_uniform_spacing_and_get_delta(vector):
    diff = np.diff(vector)
    if not np.all(np.isclose(diff[0], diff)):
        raise ValueError(f'Vector {diff} is not uniformly spaced.')
    return diff[0]


def main(args):

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)

    logger.info("Opening local cube zarr file: {}".format(args.cube_path))
    cube = xr.open_zarr(args.cube_path, consolidated=False)

    output = xr.Dataset(
        {
            "time": cube['time'],
            "normalized_weights": normalized_latitude_weights(cube) * xr.ones_like(cube['area'])
        }
    )
    output.to_zarr(args.output_path, mode="w")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Climatology")
    parser.add_argument(
        "--cube-path",
        metavar="KEY",
        type=str,
        action="store",
        dest="cube_path",
        default="../cube_v4.zarr",
        help="Cube path",
    )
    parser.add_argument(
        "--output-path",
        metavar="KEY",
        type=str,
        action="store",
        dest="output_path",
        default="lat_weights.zarr",
        help="Output path",
    )
    parser.set_defaults(log_target_var=False)    
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    args = parser.parse_args()

    main(args)
