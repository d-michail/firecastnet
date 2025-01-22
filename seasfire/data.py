from typing import Optional, List
import numpy as np
import xarray as xr
import xbatcher
from torch.utils.data import Dataset, DataLoader, RandomSampler
import logging
import tqdm
import json
import lightning as L

logger = logging.getLogger(__name__)


class SeasFireDataModule(L.LightningDataModule):
    def __init__(
        self,
        cube_path: str = "../1d_small_cube_v3.zarr",
        input_vars: List[str] = [
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
        ],
        static_vars: List[str] = ["lsm"],
        generate_only_positive_samples: bool = True,
        oci_enabled: bool = False,
        oci_input_vars: List[str] = [],
        oci_lag: int = 10,
        climatology_enabled: bool = False,
        climatology_cube: str = "climatology.zarr",
        climatology_var_name: str = "clim_gwis_ba_octodays_mean",
        log_preprocess_input_vars: List[str] = ["tp", "pop_dens"],
        target_var="gwis_ba",
        target_shift=1,
        target_var_per_area=False,
        target_var_log_process=False,
        timeseries_weeks=1,
        train_start="2002-01-01",
        train_end="2018-01-01",
        val_start="2018-01-01",
        val_end="2019-01-01",
        test_start="2019-01-01",
        test_end="2020-01-01",
        lat_dim=None,
        lon_dim=None,
        lat_dim_overlap: int = None,
        lon_dim_overlap: int = None,
        time_dim_overlap: int = None,
        task: str = "classification",
        batch_size: int = 1,
        num_workers: int = 1,
        pin_memory=False,
        load_cube_in_memory=True,
        train_random_sample=None,
        mean_std_dict_prefix: str = "cube",
    ):
        super().__init__()

        self._cube_path = cube_path
        logger.info("Cube path={}".format(self._cube_path))

        logger.info("Opening local cube zarr file: {}".format(self._cube_path))
        self._cube = xr.open_zarr(self._cube_path, consolidated=False)
        if load_cube_in_memory:
            logger.info("Loading the whole cube in memory.")
            self._cube.load()

        logger.info("Cube: {}".format(self._cube))
        logger.info("Vars: {}".format(self._cube.data_vars))

        logger.info(
            "Cube spatial dimensions: {}x{}".format(
                self._cube["latitude"].size, self._cube["longitude"].size
            )
        )

        if lat_dim is None:
            self._lat_dim = self._cube["latitude"].size
        else:
            if lat_dim <= 0 or lat_dim > self._cube["latitude"].size:
                raise ValueError("Invalid lat_dim")
            self._lat_dim = lat_dim

        if lon_dim is None:
            self._lon_dim = self._cube["longitude"].size
        else:
            if lon_dim <= 0 or lon_dim > self._cube["longitude"].size:
                raise ValueError("Invalid lon_dim")
            self._lon_dim = lon_dim

        self._lat_dim_overlap = lat_dim_overlap
        if lat_dim_overlap is not None:
            logger.info("Using latitude overlap {} in samples".format(lat_dim_overlap))
        self._lon_dim_overlap = lon_dim_overlap
        if lon_dim_overlap is not None:
            logger.info("Using longitude overlap {} in samples".format(lon_dim_overlap))

        self._time_dim_overlap = time_dim_overlap
        if self._time_dim_overlap is None:
            logger.info("Using time overlap {} in samples".format(timeseries_weeks - 1))
        else:
            logger.info(
                "Using time overlap {} in samples".format(self._time_dim_overlap)
            )

        logger.debug("Latitude: {}".format(self._cube["latitude"]))
        logger.debug("Longitude: {}".format(self._cube["longitude"]))

        self._log_preprocess_input_vars = log_preprocess_input_vars
        for var_name in self._log_preprocess_input_vars:
            logger.info("Log-transforming input var: {}".format(var_name))
            self._cube[var_name] = xr.DataArray(
                np.log(1.0 + self._cube[var_name].values),
                coords=self._cube[var_name].coords,
                dims=self._cube[var_name].dims,
                attrs=self._cube[var_name].attrs,
            )

        self._input_vars = input_vars
        for input_var in self._input_vars:
            logger.debug(
                "Var name {}, description: {}".format(
                    input_var, self._cube[input_var].description
                )
            )

        self._generate_only_positive_samples = generate_only_positive_samples
        logger.info(f"Will only generate positive samples={generate_only_positive_samples}.")

        self._oci_enabled = oci_enabled
        logger.info(f"OCI variables enabled={oci_enabled}.")

        self._oci_input_vars = oci_input_vars
        for oci_var in self._oci_input_vars:
            logger.debug(
                "Oci name {}, description: {}".format(
                    oci_var, self._cube[oci_var].description
                )
            )
        self._oci_lag = oci_lag

        self._target_var_per_area = target_var_per_area
        if self._target_var_per_area:
            logger.info("Expanding time dimension on area variable.")
            logger.info("Will compute target variable per area.")
            self._cube["area"] = self._cube["area"].expand_dims(
                dim={"time": self._cube.time}, axis=0
            )
            logger.info("Converting area in hectars")
            self._cube["area"] = self._cube["area"] / 10000.0

        self._target_var_log_process = target_var_log_process

        self._static_vars = static_vars
        for static_v in static_vars:
            logger.info(
                "Expanding time dimension on static variable = {}.".format(static_v)
            )
            self._cube[static_v] = self._cube[static_v].expand_dims(
                dim={"time": self._cube.time}, axis=0
            )

        self._clima = None
        if climatology_enabled: 
            logger.info("Opening climatology cube")
            clima_cube = xr.open_zarr(climatology_cube, consolidated=False)
            self._clima = clima_cube[climatology_var_name]
            self._clima.load()
            logger.info("Using variable {}".format(climatology_var_name))
            clima_cube.close()

        self._timeseries_weeks = timeseries_weeks

        self._target_var = target_var
        self._target_shift = target_shift
        self._train_start = train_start
        self._train_end = train_end
        self._val_start = val_start
        self._val_end = val_end
        self._test_start = test_start
        self._test_end = test_end
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._train_random_sample = train_random_sample

        self._mean_std_dict = None
        self._mean_std_dict_prefix = mean_std_dict_prefix
        self._data_train: Optional[Dataset] = None
        self._data_val: Optional[Dataset] = None
        self._data_test: Optional[Dataset] = None
        self._task = task

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_batches, train_oci_batches, self._mean_std_dict = sample_dataset(
                self._cube.copy(),
                self._input_vars,
                self._static_vars,
                self._oci_input_vars,
                self._target_var,
                self._target_shift,
                self._train_start,
                self._train_end,
                self._val_start, 
                self._val_end, 
                self._test_start, 
                self._test_end,
                self._target_var_per_area,
                self._target_var_log_process,
                self._timeseries_weeks,
                "train",
                self._lat_dim,
                self._lon_dim,
                self._lat_dim_overlap,
                self._lon_dim_overlap,
                self._time_dim_overlap,
                generate_only_positive_samples=self._generate_only_positive_samples,
                oci_enabled=self._oci_enabled,
                oci_lag=self._oci_lag,
            )

            # Save mean std dict for normalization during inference time
            logger.info("mean_std_dist={}".format(self._mean_std_dict))
            with open(
                f"{self._mean_std_dict_prefix}_mean_std_dict_{self._target_shift}.json",
                "w",
            ) as f:
                f.write(json.dumps(self._mean_std_dict))

            val_batches, val_oci_batches, _ = sample_dataset(
                self._cube.copy(),
                self._input_vars,
                self._static_vars,
                self._oci_input_vars,
                self._target_var,
                self._target_shift,
                self._train_start,
                self._train_end,                
                self._val_start,
                self._val_end, 
                self._test_start,                
                self._test_end,
                self._target_var_per_area,
                self._target_var_log_process,
                self._timeseries_weeks,
                "val",
                self._lat_dim,
                self._lon_dim,
                self._lat_dim_overlap,
                self._lon_dim_overlap,
                self._time_dim_overlap,
                generate_only_positive_samples=self._generate_only_positive_samples,
                oci_enabled=self._oci_enabled,
                oci_lag=self._oci_lag,
            )
            self._data_train = BatcherDataset(
                self._cube.copy(),
                train_batches,
                self._input_vars,
                self._static_vars,
                train_oci_batches,
                self._oci_input_vars,
                self._clima,
                self._target_var,
                self._mean_std_dict,
                self._oci_lag,
                self._oci_enabled,
                task=self._task,
                keep_only_last_timestep=False,
            )
            self._data_val = BatcherDataset(
                self._cube.copy(),
                val_batches,
                self._input_vars,
                self._static_vars,
                val_oci_batches,
                self._oci_input_vars,
                self._clima,
                self._target_var,
                self._mean_std_dict,
                self._oci_lag,
                self._oci_enabled,
                task=self._task,
                keep_only_last_timestep=False,
            )

        if stage == "test" or stage is None:
            if self._mean_std_dict is None:
                logger.info("Loading mean and std dictionary used during training")
                with open(
                    f"{self._mean_std_dict_prefix}_mean_std_dict_{self._target_shift}.json",
                    "r",
                ) as f:
                    self._mean_std_dict = json.load(f)
                    logger.info("mean_std_dict={}".format(self._mean_std_dict))

            test_batches, test_oci_batches, _ = sample_dataset(
                self._cube.copy(),
                self._input_vars,
                self._static_vars,
                self._oci_input_vars,
                self._target_var,
                self._target_shift,
                self._train_start,
                self._train_end,
                self._val_start,
                self._val_end, 
                self._test_start,
                self._test_end,
                self._target_var_per_area,
                self._target_var_log_process,
                self._timeseries_weeks,
                "test",
                self._lat_dim,
                self._lon_dim,
                self._lat_dim_overlap,
                self._lon_dim_overlap,
                None,
                generate_only_positive_samples=self._generate_only_positive_samples,                
                oci_enabled=self._oci_enabled,
                oci_lag=self._oci_lag,
            )

            self._data_test = BatcherDataset(
                self._cube.copy(),
                test_batches,
                self._input_vars,
                self._static_vars,
                test_oci_batches,
                self._oci_input_vars,
                self._clima,
                self._target_var,
                self._mean_std_dict,
                self._oci_lag,
                self._oci_enabled,
                task=self._task,
                keep_only_last_timestep=False,
            )

    def train_dataloader(self):
        if self._data_train is None:
            raise ValueError("Setup needs to be called first")

        if not self._train_random_sample is None:
            if isinstance(self._train_random_sample, int):
                num_samples = self._train_random_sample
            elif isinstance(self._train_random_sample, float):
                num_samples = int(self._train_random_sample * len(self._data_train))
            else:
                raise ValueError(
                    "Invalid type for train_random_sample. Expected int or float."
                )

            logger.info(
                f"Enabling random sample with replacement, num_samples={num_samples}."
            )
            sampler = RandomSampler(
                self._data_train, replacement=True, num_samples=num_samples
            )
            return DataLoader(
                dataset=self._data_train,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
                sampler=sampler,
                persistent_workers=True,
            )
        else:
            return DataLoader(
                dataset=self._data_train,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
                shuffle=True,
                persistent_workers=True,
            )

    def val_dataloader(self):
        if self._data_val is None:
            raise ValueError("Setup needs to be called first")

        return DataLoader(
            dataset=self._data_val,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        if self._data_test is None:
            raise ValueError("Setup needs to be called first")

        return DataLoader(
            dataset=self._data_test,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

# the code is taken from https://github.com/google-deepmind/graphcast/blob/main/graphcast/losses.py



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


def sample_dataset(
    ds,
    input_vars,
    static_vars,
    oci_input_vars,
    target_var,
    target_shift,
    train_start,
    train_end,
    val_start,
    val_end,
    test_start,
    test_end,
    target_var_per_area,
    target_var_log_process,
    timeseries_weeks,
    split,
    dim_lat,
    dim_lon,
    dim_lat_overlap=None,
    dim_lon_overlap=None,
    dim_time_overlap=None,
    load_ds_in_memory=True,
    generate_only_positive_samples=True,
    oci_enabled=False,
    oci_lag=10,
):
    logger.info("Sampling dataset for split = {}".format(split))

    logger.info(f"Shifting inputs by {target_shift}.")

    for var in input_vars:
        if target_shift > 0:
            ds[var] = ds[var].shift(time=target_shift, fill_value=0)

    # to get the weights on the seasfire dataset you use the following line
    ds.assign(normalized_weights = normalized_latitude_weights(ds) * xr.ones_like(ds['area']))

    if target_var_per_area:
        logger.info("Converting target to target per area")
        ds[target_var] = ds[target_var] / ds["area"]

    if target_var_log_process: 
        logger.info("Computing logarithm of target var")
        # ds[target_var] = np.log1p(ds[target_var])
        ds[target_var] = np.log(1.0 + ds[target_var])

    oci_ds = None
    if oci_enabled and len(oci_input_vars) > 0:
        logger.info(f"Shifting oci inputs by {target_shift}.")
        for var in oci_input_vars:
            if target_shift > 0:
                ds[var] = ds[var].shift(time=target_shift, fill_value=0)

        logger.info(f"Building oci dataset with vars={oci_input_vars}.")
        oci_ds = xr.Dataset()
        for var in oci_input_vars:
            # resample var to 1 month
            oci_ds[var] = ds[var].fillna(0).resample(time="1ME").mean(dim="time")
        if load_ds_in_memory:
            oci_ds.load()
        if "oci_pdo" in oci_input_vars:
            oci_ds["oci_pdo"] = (
                oci_ds["oci_pdo"].where(oci_ds["oci_pdo"] > -9).ffill(dim="time")
            )
        if "oci_epo" in oci_input_vars:
            oci_ds["oci_epo"] = (
                oci_ds["oci_epo"].where(oci_ds["oci_epo"] > -90).ffill(dim="time")
            )

    if split == "train":
        ds = ds.sel(time=slice(train_start, train_end))
        logger.info("train from {} to {}".format(train_start, train_end))
    elif split == "val":
        ds = ds.sel(time=slice(val_start, val_end))
        logger.info("validate from {} to {}".format(val_start, val_end))
    elif split == "test":
        ds = ds.sel(time=slice(test_start, test_end))
        logger.info("test from {} to {}".format(test_start, test_end))
    else:
        raise ValueError("Invalid split")
    # if split == "train":
    #     ds = ds.sel(time=slice("2002-01-01", "2018-01-01"))
    # elif split == "val":
    #     ds = ds.sel(time=slice("2018-01-01", "2019-01-01"))
    # elif split == "test":
    #     ds = ds.sel(time=slice("2019-01-01", "2020-01-01"))
    # else:
    #     raise ValueError("Invalid split")

    logger.info("Creating split: {}".format(split))
    logger.info("Using data in [{}, {})".format(ds.time.values[0], ds.time.values[-1]))

    ds = ds[input_vars + static_vars + [target_var]]

    input_overlap = {}
    if dim_time_overlap is None:
        if timeseries_weeks - 1 > 0:
            input_overlap["time"] = timeseries_weeks - 1
    else:
        input_overlap["time"] = dim_time_overlap
    if dim_lat_overlap is not None:
        input_overlap["latitude"] = dim_lat_overlap
    if dim_lon_overlap is not None:
        input_overlap["longitude"] = dim_lon_overlap

    if load_ds_in_memory:
        ds.load()

    bgen = xbatcher.BatchGenerator(
        ds=ds,
        input_dims={
            "longitude": dim_lon,
            "latitude": dim_lat,
            "time": timeseries_weeks,
        },
        input_overlap=input_overlap,
    )

    mean_std_dict = {}
    for var in input_vars + static_vars + [target_var]:
        mean_std_dict[var + "_mean"] = ds[var].mean().values.item(0)
        mean_std_dict[var + "_std"] = ds[var].std().values.item(0)

    if oci_enabled:
        for var in oci_input_vars:
            mean_std_dict[var + "_mean"] = oci_ds[var].mean().values.item(0)
            mean_std_dict[var + "_std"] = oci_ds[var].std().values.item(0)

    n_batches = 0
    n_pos = 0
    batches = []
    oci_batches = []
    for batch in tqdm.tqdm(bgen):
        is_positive = batch[target_var].sum() > 0
        if is_positive: 
            n_pos += 1

        if not generate_only_positive_samples or is_positive:
            batches.append(batch)
            n_batches += 1

            if oci_enabled and oci_ds is not None:
                oci_batch = oci_ds.sel(
                    time=slice(
                        batch.time[-1] - np.timedelta64(oci_lag * 31, "D"),
                        batch.time[-1],
                    )
                )
                oci_batches.append(oci_batch)


    logger.info("# of batches = {}".format(n_batches))
    logger.info("# of batches with positives = {}".format(n_pos))

    return batches, oci_batches, mean_std_dict


class BatcherDataset(Dataset):
    """Dataset from Xbatcher"""

    def __init__(
        self,
        ds,
        batches,
        input_vars,
        static_vars,
        oci_batches,
        input_oci_vars,
        clima,
        target_var,
        mean_std_dict,
        oci_lag,
        oci_enabled,
        task="classification",
        keep_only_last_timestep=True,
    ):
        self.ds = ds
        self.batches = batches
        self.input_vars = input_vars
        self.static_vars = static_vars
        self.oci_batches = oci_batches
        self.input_oci_vars = input_oci_vars
        self.clima = clima
        self.target_var = target_var
        self.mean_std_dict = mean_std_dict
        self.mean = np.stack(
            [
                mean_std_dict[f"{var}_mean"]
                for var in input_vars
                + static_vars
                + (input_oci_vars if oci_enabled else [])
            ]
        )
        self.std = np.stack(
            [
                mean_std_dict[f"{var}_std"]
                for var in input_vars
                + static_vars
                + (input_oci_vars if oci_enabled else [])
            ]
        )
        self.oci_lag = oci_lag
        self.oci_enabled = oci_enabled
        self.task = task
        self.keep_only_last_timestep = keep_only_last_timestep

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        if self.keep_only_last_timestep:
            batch = self.batches[idx].isel(time=-1)
        else:
            batch = self.batches[idx]

        inputs = np.stack(
            [batch[var] for var in self.input_vars + self.static_vars]
        ).astype(np.float32)
        for i, var in enumerate(self.input_vars):
            inputs[i] = (
                inputs[i] - self.mean_std_dict[f"{var}_mean"]
            ) / self.mean_std_dict[f"{var}_std"]

        if self.oci_enabled and len(self.input_oci_vars) > 0:
            oci_batch = self.oci_batches[idx].isel(time=slice(-self.oci_lag, None))
            oci_inputs = np.stack([oci_batch[var] for var in self.input_oci_vars])
            for i, var in enumerate(self.input_oci_vars):
                oci_inputs[i] = (
                    oci_inputs[i] - self.mean_std_dict[f"{var}_mean"]
                ) / self.mean_std_dict[f"{var}_std"]

        target = batch[self.target_var].values
        target = np.expand_dims(target, axis=0)
        inputs = np.nan_to_num(inputs, nan=-1)
        target = np.nan_to_num(target, nan=0)
        if self.task == "classification":
            target = np.where(target != 0, 1, 0)

        if self.clima is not None:
            clima_var = self.clima.sel(latitude=batch.latitude,longitude=batch.longitude,time=batch.time).values

        result = {}
        result["x"] = inputs
        if self.oci_enabled and len(self.input_oci_vars) > 0: 
            result["oci"] = oci_inputs
        result["y"] = target
        if self.clima is not None:
            result["clima"] = clima_var
            
        return result
