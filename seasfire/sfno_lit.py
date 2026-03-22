import lightning as L
import logging
import numpy as np
import torch
import torch.nn as nn
import xarray as xr

from torchmetrics import (
    AUROC,
    AveragePrecision,
    CriticalSuccessIndex,
    F1Score,
    MeanSquaredError,
    MeanAbsoluteError,
    R2Score,
    SymmetricMeanAbsolutePercentageError,
)

from .backbones.loss.fcn_cls_loss import FCNClassificationLoss
from .backbones.loss.regression_area_loss import (
    CellAreaWeightedHuberLossFunction,
    CellAreaWeightedL1LossFunction,
    CellAreaWeightedMSELossFunction,
)
from .backbones.graphcast.graph_utils import deg2rad, grid_cell_area
from .backbones.sfno_net import SFNONet

logger = logging.getLogger(__name__)


class SFNOLit(L.LightningModule):
    """Lightning module wrapping the Spherical Fourier Neural Operator (SFNO) backbone.

    Temporal context is handled entirely by the CubeEmbedder (3-D convolution), which
    compresses the time dimension before passing the single-timestep representation to
    the SFNO.  The SFNO operates on a regular lat-lon grid at the downsampled resolution
    (default 1°/180×360) and the result is upsampled back to the input resolution via
    PixelShuffle.

    Feature parity with FireCastNetLit: LSM masking, GFED region loss weighting,
    area-weighted regression losses, static positional features, and both
    classification and regression tasks are all supported.
    """

    def __init__(
        self,
        sp_res: float = 0.250,
        max_lat: float = 89.875,
        min_lat: float = -89.875,
        max_lon: float = 179.875,
        min_lon: float = -179.875,
        lat_lon_static_data: bool = True,
        embed_cube: bool = True,
        embed_cube_width: int = 4,
        embed_cube_height: int = 4,
        embed_cube_time: int = 3,
        embed_cube_dim: int = 32,
        embed_cube_layer_norm: bool = False,
        embed_cube_vit_enable: bool = False,
        embed_cube_vit_patch_size: int = 36,
        embed_cube_vit_dim: int = 64,
        embed_cube_vit_depth: int = 1,
        embed_cube_vit_heads: int = 1,
        embed_cube_vit_mlp_dim: int = 64,
        embed_cube_ltae_enable: bool = False,
        embed_cube_ltae_num_heads: int = 4,
        embed_cube_ltae_d_k: int = 16,
        timeseries_len: int = 1,
        input_dim_grid_nodes: int = 11,
        output_dim_grid_nodes: int = 16,
        sfno_embed_dim: int = 128,
        sfno_num_layers: int = 4,
        sfno_scale_factor: int = 3,
        sfno_hard_thresholding_fraction: float = 1.0,
        sfno_normalization_layer: str = "instance_norm",
        sfno_use_mlp: bool = True,
        sfno_mlp_ratio: float = 2.0,
        sfno_pos_embed: str = "learnable latlon",
        task: str = "classification",
        regression_loss: str = "mse",
        cube_path: str = "cube.zarr",
        gfed_region_enable_loss_weighting: bool = False,
        gfed_region_var_name: str = "gfed_region",
        gfed_region_weights=None,
        val_gfed_regions: list[str] | None = None,
        test_gfed_regions: list[str] | None = None,
        lsm_filter_enable: bool = True,
        lsm_var_name: str = "lsm",
        lsm_threshold: float = 0.05,
        lr: float = 0.001,
        weight_decay: float = 0.00000001,
        max_epochs: int = 100,
        optimizer_apex: bool = False,
        optimizer_fused: bool = False,
        display_model_example: bool = True,
        display_model_example_precision: str = "bf16-true",
    ):
        super().__init__()

        self.save_hyperparameters()

        self._init_lat_lon(
            sp_res,
            max_lat,
            min_lat,
            max_lon,
            min_lon,
            lat_lon_static_data,
            input_dim_grid_nodes,
        )

        self._cube_path = cube_path

        self._init_gfed_regions(
            gfed_region_enable_loss_weighting,
            gfed_region_var_name,
            gfed_region_weights,
            val_gfed_regions,
            test_gfed_regions,
        )

        self.register_buffer("_lsm_mask", torch.empty(0), persistent=True)
        self._init_lsm_filter(
            lsm_filter_enable,
            lsm_var_name,
            lsm_threshold,
        )

        self._lr = lr
        self._max_epochs = max_epochs
        self._weight_decay = weight_decay
        self._optimizer_apex = optimizer_apex
        self._optimizer_fused = optimizer_fused

        self._init_metrics(task, regression_loss)

        self._init_net(
            timeseries_len,
            output_dim_grid_nodes,
            embed_cube,
            embed_cube_width,
            embed_cube_height,
            embed_cube_time,
            embed_cube_dim,
            embed_cube_layer_norm,
            embed_cube_vit_enable,
            embed_cube_vit_patch_size,
            embed_cube_vit_dim,
            embed_cube_vit_depth,
            embed_cube_vit_heads,
            embed_cube_vit_mlp_dim,
            embed_cube_ltae_enable,
            embed_cube_ltae_num_heads,
            embed_cube_ltae_d_k,
            sfno_embed_dim,
            sfno_num_layers,
            sfno_scale_factor,
            sfno_hard_thresholding_fraction,
            sfno_normalization_layer,
            sfno_use_mlp,
            sfno_mlp_ratio,
            sfno_pos_embed,
        )

        self._init_example(
            timeseries_len, display_model_example, display_model_example_precision
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_lat_lon(
        self,
        sp_res,
        max_lat,
        min_lat,
        max_lon,
        min_lon,
        lat_lon_static_data: bool,
        input_dim_grid_nodes: int,
    ):
        self._input_dim_grid_nodes = input_dim_grid_nodes
        logger.info("Using input dimensions = {}".format(self._input_dim_grid_nodes))

        self._latitudes = torch.from_numpy(
            np.arange(max_lat, min_lat - (sp_res / 2), -sp_res, dtype=np.float32)
        )
        self._lat_dim = self._latitudes.size(0)
        logger.info("Latitude dimension = {}".format(self._lat_dim))

        self._longitudes = torch.from_numpy(
            np.arange(min_lon, max_lon + (sp_res / 2), sp_res, dtype=np.float32)
        )
        self._lon_dim = self._longitudes.size(0)
        logger.info("Longitude dimension = {}".format(self._lon_dim))

        self._lat_lon_grid = torch.stack(
            torch.meshgrid(self._latitudes, self._longitudes, indexing="ij"), dim=-1
        )

        self._area = grid_cell_area(self._lat_lon_grid[:, :, 0], unit="deg")

        self._static_data = None
        if lat_lon_static_data:
            logger.info("Will use static data (cos_lat, sin_lon and cos_lon)")

            cos_lat = torch.cos(deg2rad(self._latitudes))
            cos_lat = cos_lat.view(1, 1, self._latitudes.size(0), 1)
            cos_lat_mg = cos_lat.expand(
                1, 1, self._latitudes.size(0), self._longitudes.size(0)
            )

            sin_lon = torch.sin(deg2rad(self._longitudes))
            sin_lon = sin_lon.view(1, 1, 1, self._longitudes.size(0))
            sin_lon_mg = sin_lon.expand(
                1, 1, self._latitudes.size(0), self._longitudes.size(0)
            )

            cos_lon = torch.cos(deg2rad(self._longitudes))
            cos_lon = cos_lon.view(1, 1, 1, self._longitudes.size(0))
            cos_lon_mg = cos_lon.expand(
                1, 1, self._latitudes.size(0), self._longitudes.size(0)
            )

            self._static_data = torch.cat((cos_lat_mg, sin_lon_mg, cos_lon_mg), dim=1)

            self._input_dim_grid_nodes = self._input_dim_grid_nodes + 3
            logger.info(
                "Increased input dimensions to {}".format(self._input_dim_grid_nodes)
            )

    def _init_gfed_regions(
        self,
        gfed_region_enable_loss_weighting,
        gfed_region_var_name,
        gfed_region_weights,
        val_gfed_regions,
        test_gfed_regions,
        dtype=torch.float32,
    ):
        self._gfed_region_enable_loss_weighting = gfed_region_enable_loss_weighting
        self._val_gfed_region_mask = None
        self._test_gfed_region_mask = None

        if val_gfed_regions is None:
            val_gfed_regions = []
        if test_gfed_regions is None:
            test_gfed_regions = []

        logger.info("Validation GFED regions: {}".format(val_gfed_regions))
        self._val_gfed_regions = [self._map_region_to_int(r) for r in val_gfed_regions]

        logger.info("Test GFED regions: {}".format(test_gfed_regions))
        self._test_gfed_regions = [self._map_region_to_int(r) for r in test_gfed_regions]

        logger.info("Opening cube zarr file: {}".format(self._cube_path))
        cube = xr.open_zarr(self._cube_path, consolidated=False)
        gfed_region = cube[gfed_region_var_name].values
        gfed_region = torch.tensor(gfed_region, dtype=dtype)
        cube.close()

        if len(self._val_gfed_regions) > 0:
            self._val_gfed_region_mask = torch.isin(
                gfed_region,
                torch.tensor(self._val_gfed_regions, dtype=gfed_region.dtype),
            )

        if len(self._test_gfed_regions) > 0:
            self._test_gfed_region_mask = torch.isin(
                gfed_region,
                torch.tensor(self._test_gfed_regions, dtype=gfed_region.dtype),
            )

        if not gfed_region_enable_loss_weighting:
            return

        logger.info("Enabling GFED region weighting")

        weight_map = torch.zeros_like(gfed_region, dtype=dtype)
        for region_name, weight in gfed_region_weights.items():
            region_int = self._map_region_to_int(region_name)
            weight_map = torch.where(gfed_region == region_int, weight, weight_map)

        self._gfed_region_weights = weight_map.unsqueeze(0)

        logger.info(
            "GFED regions tensor with shape: {}".format(self._gfed_region_weights)
        )

    def _map_region_to_int(self, region: str) -> int:
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
        return region_name_to_int.get(region, -1)

    def _init_lsm_filter(
        self,
        lsm_filter_enable,
        lsm_var_name,
        lsm_threshold,
        dtype=torch.float32,
    ):
        self._lsm_filter_enable = lsm_filter_enable
        self._lsm_threshold = lsm_threshold

        if not lsm_filter_enable:
            self._lsm_mask = None
            return

        logger.info("Enabling LSM filter/mask")

        logger.info("Opening cube zarr file: {}".format(self._cube_path))
        cube = xr.open_zarr(self._cube_path, consolidated=False)
        lsm_filter = cube[lsm_var_name].values
        lsm_filter = torch.tensor(lsm_filter, dtype=dtype).unsqueeze(0)
        lsm_mask = lsm_filter < lsm_threshold
        self._lsm_mask = lsm_mask
        cube.close()

        logger.info("LSM filter/mask tensor with shape: {}".format(self._lsm_mask))

    def _init_metrics(self, task, regression_loss):
        self._task = task

        if task == "classification":
            self._criterion = FCNClassificationLoss(
                pixel_weights=(
                    self._gfed_region_weights
                    if self._gfed_region_enable_loss_weighting
                    else None
                ),
            )
            self._metrics_names = ["auc", "f1", "auprc"]
            self._val_metrics = nn.ModuleList(
                [
                    AUROC(task="binary"),
                    F1Score(task="binary"),
                    AveragePrecision(task="binary"),
                ]
            )
            self._test_metrics = nn.ModuleList(
                [
                    AUROC(task="binary"),
                    F1Score(task="binary"),
                    AveragePrecision(task="binary"),
                ]
            )
        elif task == "regression":
            if regression_loss == "l1":
                self._criterion = CellAreaWeightedL1LossFunction(self._area)
            elif regression_loss == "huber":
                self._criterion = CellAreaWeightedHuberLossFunction(self._area)
            else:
                self._criterion = CellAreaWeightedMSELossFunction(self._area)

            self._metrics_names = ["mse", "mae", "r2", "smape", "csi"]
            self._val_metrics = nn.ModuleList(
                [
                    MeanSquaredError(),
                    MeanAbsoluteError(),
                    R2Score(),
                    SymmetricMeanAbsolutePercentageError(),
                    CriticalSuccessIndex(0.5),
                ]
            )
            self._test_metrics = nn.ModuleList(
                [
                    MeanSquaredError(),
                    MeanAbsoluteError(),
                    R2Score(),
                    SymmetricMeanAbsolutePercentageError(),
                    CriticalSuccessIndex(0.5),
                ]
            )
        else:
            raise ValueError("Invalid task")

        self._metrics = {
            "val": (self._val_metrics, self._metrics_names),
            "test": (self._test_metrics, self._metrics_names),
        }

    def _init_net(
        self,
        timeseries_len,
        output_dim_grid_nodes,
        embed_cube,
        embed_cube_width,
        embed_cube_height,
        embed_cube_time,
        embed_cube_dim,
        embed_cube_layer_norm,
        embed_cube_vit_enable,
        embed_cube_vit_patch_size,
        embed_cube_vit_dim,
        embed_cube_vit_depth,
        embed_cube_vit_heads,
        embed_cube_vit_mlp_dim,
        embed_cube_ltae_enable,
        embed_cube_ltae_num_heads,
        embed_cube_ltae_d_k,
        sfno_embed_dim,
        sfno_num_layers,
        sfno_scale_factor,
        sfno_hard_thresholding_fraction,
        sfno_normalization_layer,
        sfno_use_mlp,
        sfno_mlp_ratio,
        sfno_pos_embed,
    ):
        self._net = SFNONet(
            grid_width=self._lat_dim,
            grid_height=self._lon_dim,
            timeseries_len=timeseries_len,
            input_dim_grid_nodes=self._input_dim_grid_nodes,
            output_dim_grid_nodes=output_dim_grid_nodes,
            embed_cube=embed_cube,
            embed_cube_width=embed_cube_width,
            embed_cube_height=embed_cube_height,
            embed_cube_time=embed_cube_time,
            embed_cube_dim=embed_cube_dim,
            embed_cube_layer_norm=embed_cube_layer_norm,
            embed_cube_vit_enable=embed_cube_vit_enable,
            embed_cube_vit_patch_size=embed_cube_vit_patch_size,
            embed_cube_vit_dim=embed_cube_vit_dim,
            embed_cube_vit_depth=embed_cube_vit_depth,
            embed_cube_vit_heads=embed_cube_vit_heads,
            embed_cube_vit_mlp_dim=embed_cube_vit_mlp_dim,
            embed_cube_ltae_enable=embed_cube_ltae_enable,
            embed_cube_ltae_num_heads=embed_cube_ltae_num_heads,
            embed_cube_ltae_d_k=embed_cube_ltae_d_k,
            sfno_embed_dim=sfno_embed_dim,
            sfno_num_layers=sfno_num_layers,
            sfno_scale_factor=sfno_scale_factor,
            sfno_hard_thresholding_fraction=sfno_hard_thresholding_fraction,
            sfno_normalization_layer=sfno_normalization_layer,
            sfno_use_mlp=sfno_use_mlp,
            sfno_mlp_ratio=sfno_mlp_ratio,
            sfno_pos_embed=sfno_pos_embed,
        )

    def _init_example(
        self,
        timeseries_len,
        display_model_example: bool,
        display_model_example_precision: str,
    ):
        if not display_model_example:
            logger.info("Skipping model example generation")
            self.example_input_array = None
            return

        logger.info("Will create a model example")

        if display_model_example_precision in ("bf16-true", "bf16-mixed"):
            dtype = torch.bfloat16
        elif display_model_example_precision in ("16-true", "16-mixed"):
            dtype = torch.float16
        else:
            dtype = torch.float32

        logger.info("Using dtype={} for example data".format(dtype))

        self.example_input_array = torch.empty(
            (
                self._input_dim_grid_nodes,
                timeseries_len,
                self._lat_dim,
                self._lon_dim,
            ),
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(self, x, oci, y):
        if len(x.size()) != 5:
            raise ValueError("Model accepts input of shape [1, C, T, W, H]")

        if self._static_data is not None:
            static = self._static_data
            if len(static.size()) == 4:
                timesteps = x.size(2)
                static = static.unsqueeze(2).repeat(1, 1, timesteps, 1, 1).to(x)
            x = torch.cat((x, static), dim=1)

        batches = x.size(0)
        if batches != 1:
            raise ValueError("No support for batch size > 1")
        x = x[0]

        if oci is not None:
            if oci.size(0) != 1:
                raise ValueError("No support for batch size > 1")
            oci = oci[0]

        if y is not None:
            batches = y.size(0)
            if batches != 1:
                raise ValueError("No support for batch size > 1")
            y = y[0]

        return x, oci, y

    # ------------------------------------------------------------------
    # Lightning interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, oci: torch.Tensor = None):
        return self._net(x)

    def training_step(self, batch, batch_idx):
        x = batch.get("x")
        oci = batch.get("oci")
        y = batch.get("y")

        x, oci, y = self._prepare_data(x, oci, y)
        logits = self(x, oci)

        logits = logits[:, -1, :, :]
        y = y[:, -1, :, :]

        if self._task == "classification":
            loss = self._criterion(logits, y.to(torch.float32))
        else:
            loss = self._criterion(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def evaluate(self, batch, stage=None):
        x = batch.get("x")
        oci = batch.get("oci")
        y = batch.get("y")

        x, oci, y = self._prepare_data(x, oci, y)
        logits = self(x, oci)

        logits = logits[:, -1, :, :]
        y = y[:, -1, :, :]

        if self._task == "classification":
            loss = self._criterion(logits, y.to(torch.float32))
            preds = torch.sigmoid(logits)
        else:
            loss = self._criterion(logits, y)
            preds = logits

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        metrics, metrics_names = self._metrics[stage]

        if self._lsm_mask is not None:
            preds[self._lsm_mask] = 0

        preds = preds.view(-1)
        y = y.view(-1)

        if stage == "val" and self._val_gfed_region_mask is not None:
            mask = self._val_gfed_region_mask.view(-1)
            preds = preds[mask]
            y = y[mask]

        if stage == "test" and self._test_gfed_region_mask is not None:
            mask = self._test_gfed_region_mask.view(-1)
            preds = preds[mask]
            y = y[mask]

        for idx, metric in enumerate(metrics):
            name = metrics_names[idx]
            metric.update(preds, y)
            self.log(
                f"{stage}_{name}", metric, on_step=False, on_epoch=True, prog_bar=True
            )

    def validation_step(self, batch):
        self.evaluate(batch, "val")

    def test_step(self, batch):
        self.evaluate(batch, "test")

    def predict_step(self, batch):
        x = batch.get("x")
        oci = batch.get("oci")
        y = batch.get("y")

        x, oci, y = self._prepare_data(x, oci, y)
        logits = self(x, oci)

        logits = logits[:, -1, :, :]

        if self._task == "classification":
            preds = torch.sigmoid(logits)
        else:
            preds = logits

        if self._lsm_mask is not None:
            preds[self._lsm_mask] = 0

        return preds

    def configure_optimizers(self):
        optimizer = None
        if self._optimizer_apex:
            try:
                from apex.optimizers import FusedAdam

                optimizer = FusedAdam(self.parameters(), lr=self._lr, betas=(0.9, 0.95))
            except ImportError:
                logger.warning(
                    "NVIDIA Apex (https://github.com/nvidia/apex) is not installed, "
                    "FusedAdam optimizer will not be used."
                )

        if optimizer is None:
            fused = self._optimizer_fused
            if fused and any(p.is_complex() for p in self.parameters()):
                logger.warning(
                    "Disabling fused AdamW: complex-valued parameters (SFNO spectral "
                    "weights) are not supported by the fused optimizer."
                )
                fused = False
            if fused:
                logger.info("Optimizer has fused enabled")
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self._lr,
                weight_decay=self._weight_decay,
                fused=fused,
            )
        logger.info(f"Using {optimizer.__class__.__name__} optimizer")

        if self._max_epochs <= 10:
            logger.warning(f"Max epochs {self._max_epochs} should be larger than 10")
        lr_scheduler1 = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.3, end_factor=1, total_iters=10
        )
        lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self._max_epochs - 10, T_mult=1
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[lr_scheduler1, lr_scheduler2], milestones=[10]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "train_loss",
        }
