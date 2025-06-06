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

from .backbones.loss.regression_area_loss import (
    CellAreaWeightedL1LossFunction,
    CellAreaWeightedMSELossFunction,
    CellAreaWeightedHuberLossFunction,
)
from .backbones.loss.fcn_cls_loss import FCNClassificationLoss
from .backbones.graphcast.graph_utils import deg2rad, grid_cell_area
from .backbones.graphcast.graph_cast_cube_net import GraphCastCubeNet
from .backbones.graphcast.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


class FireCastNetLit(L.LightningModule):
    def __init__(
        self,
        icospheres_graph_path="icospheres/icospheres_0_1_2_3_4_5_6.json.gz",
        sp_res=0.250,
        max_lat=89.875,
        min_lat=-89.875,
        max_lon=179.875,
        min_lon=-179.875,
        lat_lon_static_data: bool = True,
        embed_cube: bool = False,
        embed_cube_width: int = 4,
        embed_cube_height: int = 4,
        embed_cube_time: int = 1,
        embed_cube_dim: int = 128,
        embed_cube_layer_norm: bool = True,
        embed_cube_vit_enable: bool = False,
        embed_cube_vit_patch_size: int = 72,
        embed_cube_vit_dim: int = 64,
        embed_cube_vit_depth: int = 1,
        embed_cube_vit_heads: int = 1,
        embed_cube_vit_mlp_dim: int = 64,
        embed_cube_ltae_enable: bool = False,
        embed_cube_ltae_num_heads: int = 4,
        embed_cube_ltae_d_k: int = 16,
        embed_cube_sp_res=1.0,
        embed_cube_max_lat=89.5,
        embed_cube_min_lat=-89.5,
        embed_cube_max_lon=179.5,
        embed_cube_min_lon=-179.5,
        timeseries_len=1,
        input_dim_grid_nodes: int = 11,
        output_dim_grid_nodes: int = 1,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        processor_layers: int = 8,
        hidden_layers: int = 1,
        hidden_dim: int = 64,
        aggregation: str = "sum",
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        task: str = "classification",
        regression_loss: str = "mse",
        cube_path: str = "cube.zarr",
        gfed_region_enable_loss_weighting: bool = False,
        gfed_region_var_name="gfed_region",
        gfed_region_weights=None,
        climatology_enable_loss: bool = False,
        climatology_lambda = 0.1,
        lsm_filter_enable=True,
        lsm_var_name: str = "lsm",
        lsm_threshold: float = 0.05,
        lr: float = 0.01,
        weight_decay: float = 0.000001,
        max_epochs: int = 100,
        optimizer_apex: bool = False,
        optimizer_fused: bool = False,
        display_model_example: bool = True,
        display_model_example_precision: str = "bf16-true",
    ):
        super().__init__()

        self.save_hyperparameters()

        self._init_graph(
            icospheres_graph_path,
            sp_res,
            max_lat,
            min_lat,
            max_lon,
            min_lon,
            lat_lon_static_data,
            embed_cube,
            embed_cube_sp_res,
            embed_cube_max_lat,
            embed_cube_min_lat,
            embed_cube_max_lon,
            embed_cube_min_lon,
            input_dim_grid_nodes,
        )

        self._cube_path = cube_path

        self._init_gfed_regions(
            gfed_region_enable_loss_weighting,
            gfed_region_var_name,
            gfed_region_weights,
        )

        self._climatology_enable_loss = climatology_enable_loss
        self._climatology_lambda = climatology_lambda

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
            input_dim_mesh_nodes,
            input_dim_edges,
            processor_layers,
            hidden_layers,
            hidden_dim,
            aggregation,
            norm_type,
            do_concat_trick,
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
        )

        self._init_example(
            timeseries_len, display_model_example, display_model_example_precision
        )

    def _init_graph(
        self,
        icospheres_graph_path,
        sp_res,
        max_lat,
        min_lat,
        max_lon,
        min_lon,
        lat_lon_static_data: bool,
        embed_cube: bool,
        embed_cube_sp_res,
        embed_cube_max_lat,
        embed_cube_min_lat,
        embed_cube_max_lon,
        embed_cube_min_lon,
        input_dim_grid_nodes,
    ):
        self._input_dim_grid_nodes = input_dim_grid_nodes
        logger.info("Using input dimensions = {}".format(self._input_dim_grid_nodes))

        g_sp_res = embed_cube_sp_res if embed_cube else sp_res
        g_max_lat = embed_cube_max_lat if embed_cube else max_lat
        g_min_lat = embed_cube_min_lat if embed_cube else min_lat
        g_max_lon = embed_cube_max_lon if embed_cube else max_lon
        g_min_lon = embed_cube_min_lon if embed_cube else min_lon

        # create the lat_lon_grid for the graph
        self._g_latitudes = torch.from_numpy(
            np.arange(
                g_max_lat, g_min_lat - (g_sp_res / 2), -g_sp_res, dtype=np.float32
            )
        )
        self._g_lat_dim = self._g_latitudes.size(0)
        logger.info("Latitude dimension (graph) = {}".format(self._g_lat_dim))

        self._g_longitudes = torch.from_numpy(
            np.arange(g_min_lon, g_max_lon + (g_sp_res / 2), g_sp_res, dtype=np.float32)
        )
        self._g_lon_dim = self._g_longitudes.size(0)
        logger.info("Longitude dimension (graph) = {}".format(self._g_lon_dim))

        self._g_lat_lon_grid = torch.stack(
            torch.meshgrid(self._g_latitudes, self._g_longitudes, indexing="ij"), dim=-1
        )

        # create graphs
        self._gb = GraphBuilder(
            icospheres_graph_path=icospheres_graph_path,
            lat_lon_grid=self._g_lat_lon_grid,
        )

        self._mesh_graph = self._gb.create_mesh_graph()
        self._g2m_graph = self._gb.create_g2m_graph()
        self._m2g_graph = self._gb.create_m2g_graph()

        if not embed_cube:
            self._latitudes = self._g_latitudes
            self._longitudes = self._g_longitudes
            self._lat_lon_grid = self._g_lat_lon_grid
        else:
            # When embed cude is true, the input has different spatial resolution from
            # the graph. Here we need the larger.
            self._latitudes = torch.from_numpy(
                np.arange(max_lat, min_lat - (sp_res / 2), -sp_res, dtype=np.float32)
            )
            self._longitudes = torch.from_numpy(
                np.arange(min_lon, max_lon + (sp_res / 2), sp_res, dtype=np.float32)
            )
            self._lat_lon_grid = torch.stack(
                torch.meshgrid(self._latitudes, self._longitudes, indexing="ij"), dim=-1
            )

        self._lat_dim = self._latitudes.size(0)
        self._lon_dim = self._longitudes.size(0)

        # compute area for area-weighted loss function
        self._area = grid_cell_area(self._lat_lon_grid[:, :, 0], unit="deg")

        # use static data or not?
        self._static_data = None
        if lat_lon_static_data:
            logger.info("Will use static data (cos_lat, sin_lon and cos_lon)")

            # cos latitudes
            cos_lat = torch.cos(deg2rad(self._latitudes))
            cos_lat = cos_lat.view(1, 1, self._latitudes.size(0), 1)
            cos_lat_mg = cos_lat.expand(
                1, 1, self._latitudes.size(0), self._longitudes.size(0)
            )

            # sin longitudes
            sin_lon = torch.sin(deg2rad(self._longitudes))
            sin_lon = sin_lon.view(1, 1, 1, self._longitudes.size(0))
            sin_lon_mg = sin_lon.expand(
                1, 1, self._latitudes.size(0), self._longitudes.size(0)
            )

            # cos longitudes
            cos_lon = torch.cos(deg2rad(self._longitudes))
            cos_lon = cos_lon.view(1, 1, 1, self._longitudes.size(0))
            cos_lon_mg = cos_lon.expand(
                1, 1, self._latitudes.size(0), self._longitudes.size(0)
            )

            self._static_data = torch.cat((cos_lat_mg, sin_lon_mg, cos_lon_mg), dim=1)

            # increase input dimension by static data
            self._input_dim_grid_nodes = self._input_dim_grid_nodes + 3
            logger.info(
                "Increased input dimensions to {}".format(self._input_dim_grid_nodes)
            )

    def _init_gfed_regions(
        self,
        gfed_region_enable_loss_weighting,
        gfed_region_var_name,
        gfed_region_weights,
        dtype=torch.float32,
    ):
        self._gfed_region_enable_loss_weighting = gfed_region_enable_loss_weighting

        if not gfed_region_enable_loss_weighting:
            return

        logger.info("Enabling GFED region weightning")

        logger.info("Opening cube zarr file: {}".format(self._cube_path))
        cube = xr.open_zarr(self._cube_path, consolidated=False)
        gfed_region = cube[gfed_region_var_name].values
        gfed_region = torch.tensor(gfed_region, dtype=dtype)
        cube.close()

        # Map GFED regions to weights
        weight_map = torch.zeros_like(gfed_region, dtype=dtype)
        for region, weight in gfed_region_weights.items():
            weight_map = torch.where(gfed_region == region, weight, weight_map)

        self._gfed_region_weights = weight_map.unsqueeze(0)

        logger.info(
            "GFED regions tensor with shape: {}".format(self._gfed_region_weights)
        )

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
        self.register_buffer("_lsm_mask", lsm_mask)
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
                enable_clima=self._climatology_enable_loss,
                clima_lambda=self._climatology_lambda
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
        input_dim_mesh_nodes: int,
        input_dim_edges: int,
        processor_layers: int,
        hidden_layers: int,
        hidden_dim: int,
        aggregation: str,
        norm_type: str,
        do_concat_trick: bool,
        timeseries_len,
        output_dim_grid_nodes,
        embed_cube: bool,
        embed_cube_width: int,
        embed_cube_height: int,
        embed_cube_time: int,
        embed_cube_dim: int,
        embed_cube_layer_norm: bool,
        embed_cube_vit_enable: bool,
        embed_cube_vit_patch_size: int,
        embed_cube_vit_dim: int,
        embed_cube_vit_depth: int,
        embed_cube_vit_heads: int,
        embed_cube_vit_mlp_dim: int,
        embed_cube_ltae_enable: bool,
        embed_cube_ltae_num_heads: int,
        embed_cube_ltae_d_k: int,
    ):
        self._net = GraphCastCubeNet(
            mesh_graph=self._mesh_graph,
            g2m_graph=self._g2m_graph,
            m2g_graph=self._m2g_graph,
            grid_width=self._lat_dim,
            grid_height=self._lon_dim,
            timeseries_len=timeseries_len,
            input_dim_grid_nodes=self._input_dim_grid_nodes,
            input_dim_mesh_nodes=input_dim_mesh_nodes,
            input_dim_edges=input_dim_edges,
            output_dim_grid_nodes=output_dim_grid_nodes,
            processor_layers=processor_layers,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            aggregation=aggregation,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
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

        if (
            display_model_example_precision == "bf16-true"
            or display_model_example_precision == "bf16-mixed"
        ):
            dtype = torch.bfloat16
        elif (
            display_model_example_precision == "16-true"
            or display_model_example_precision == "16-mixed"
        ):
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

    def _prepare_data(self, x, oci, y, clima):
        if len(x.size()) != 5:
            raise ValueError("Model accepts input of shape [1, C, T, W, H]")

        # Check if we have some static data and concat with input.
        # It should be in the format [1, C, W, H].
        if self._static_data is not None:
            if len(self._static_data.size()) == 4:
                timesteps = x.size(2)
                self._static_data = (
                    self._static_data.unsqueeze(2).repeat(1, 1, timesteps, 1, 1).to(x)
                )
            x = torch.cat((x, self._static_data), dim=1)

        # Prepare x in right format
        # [1, C, T, W, H] -> [C, T, W, H]
        batches = x.size(0)
        if batches != 1:
            raise ValueError("No support for batch size > 1")
        x = x[0]

        # Prepare oci in right format
        if oci is not None:
            if oci.size(0) != 1:
                raise ValueError("No support for batch size > 1")
            oci = oci[0]

        # Prepare y in right format
        # [1, 1, T, W, H] -> [1, T, W, H]
        if y is not None:
            batches = y.size(0)
            if batches != 1:
                raise ValueError("No support for batch size > 1")
            y = y[0]

        return x, oci, y, clima

    def forward(self, x: torch.Tensor, oci: torch.Tensor = None):
        return self._net(x)

    def training_step(self, batch, batch_idx):
        x = batch.get("x")
        oci = batch.get("oci")
        y = batch.get("y")
        clima = batch.get("clima")
        x, oci, y, clima = self._prepare_data(x, oci, y, clima)

        logits = self(x, oci)
        
        logits = logits[:, -1, :, :]
        y = y[:, -1, :, :]
        if clima is not None:
            clima = clima[:, -1, :, :]

        if self._task == "classification":
            loss = self._criterion(logits, y.to(torch.float32), clima=clima)
        else:
            loss = self._criterion(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def evaluate(self, batch, stage=None):
        x = batch.get("x")
        oci = batch.get("oci")
        y = batch.get("y")
        clima = batch.get("clima")
        x, oci, y, clima = self._prepare_data(x, oci, y, clima)

        logits = self(x, oci)

        logits = logits[:, -1, :, :]
        y = y[:, -1, :, :]
        if clima is not None:
            clima = clima[:, -1, :, :]

        if self._task == "classification":
            loss = self._criterion(logits, y.to(torch.float32), clima=clima)
            preds = torch.sigmoid(logits)
        else:
            loss = self._criterion(logits, y)
            preds = logits

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        metrics, metrics_names = self._metrics[stage]

        # if LSM mask is present, adjust prediction to zero
        if self._lsm_mask is not None:
            preds[self._lsm_mask] = 0

        preds = preds.view(-1)
        y = y.view(-1)

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
        clima = batch.get("clima")

        x, oci, y, clima = self._prepare_data(x, oci, y, clima)

        logits = self(x, oci)

        logits = logits[:, -1, :, :]

        if self._task == "classification":
            preds = torch.sigmoid(logits)
        else:
            preds = logits

        # if LSM mask is present, adjust prediction to zero
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
                logger.warn(
                    "NVIDIA Apex (https://github.com/nvidia/apex) is not installed, FusedAdam optimizer will not be used."
                )

        if optimizer is None:
            if self._optimizer_fused:
                logger.info("Optimizer has fused enabled")
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self._lr,
                weight_decay=self._weight_decay,
                fused=self._optimizer_fused,
            )
        logger.info(f"Using {optimizer.__class__.__name__} optimizer")

        if self._max_epochs <= 10:
            logger.warn(f"Max epochs {self._max_epochs} should be larger that 10")
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

    def dglTo(self, device):
        """Move all DGL graphs into a particular device."""
        self._net.dglTo(device)
