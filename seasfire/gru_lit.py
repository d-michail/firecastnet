import lightning as L
import logging
import torch
import torch.nn as nn

from torchmetrics import (
    AUROC,
    AveragePrecision,
    CriticalSuccessIndex,
    F1Score,
    SymmetricMeanAbsolutePercentageError,
)
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score

from .backbones.gru import GRUSeg

logger = logging.getLogger(__name__)


class GRULit(L.LightningModule):
    def __init__(
        self,
        input_dim_grid_nodes: int = 11,
        hidden_layers: int = 1,
        hidden_dim: int = 64,
        task: str = "classification",
        lr: float = 0.01,
        weight_decay: float = 0.000001,
        max_epochs: int = 100,
    ):
        super().__init__()

        self.save_hyperparameters()

        self._lr = lr
        self._weight_decay = weight_decay
        self._max_epochs = max_epochs

        self._init_metrics(task)

        self._init_net(
            input_dim_grid_nodes,
            hidden_layers,
            hidden_dim,
        )

    def _init_metrics(self, task):
        self._task = task

        if task == "classification":
            self._criterion = nn.BCEWithLogitsLoss()
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
            self._criterion = nn.MSELoss()
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
        hidden_layers: int,
        hidden_dim: int,
    ):
        self._net = GRUSeg(
            input_size=input_dim_mesh_nodes,
            hidden_size=hidden_dim,
            num_layers=hidden_layers,
            dropout=0.1,
            output_size=1,
        )

    def _prepare_data(self, x, y):
        if len(x.size()) != 5:
            raise ValueError("Model accepts input of shape [B, C, T, W, H]")

        # Prepare x in right format for a GRU
        # [B, C, T, W, H] -> [B * W * H, T, C]
        _, c_d, t_d, _, _ = x.size()
        x = x.permute(0, 3, 4, 2, 1)
        x = x.reshape(-1, t_d, c_d)

        # Prepare y in right format
        # [B, 1, T, W, H] -> [B * W * H, T, 1]
        y = y.permute(0, 3, 4, 2, 1)
        y = y.reshape(-1, t_d, 1)

        return x, y

    def forward(self, x: torch.Tensor):
        return self._net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = self._prepare_data(x, y)
        logits = self(x)

        y = y[:, -1, :]

        if self._task == "classification":
            loss = self._criterion(logits, y.to(torch.float32))
        else:
            loss = self._criterion(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def evaluate(self, batch, stage=None):
        x, y = batch
        x, y = self._prepare_data(x, y)
        logits = self(x)

        y = y[:, -1, :]

        if self._task == "classification":
            loss = self._criterion(logits, y.to(torch.float32))
            preds = torch.sigmoid(logits)
        else:
            loss = self._criterion(logits, y)
            preds = logits
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        metrics, metrics_names = self._metrics[stage]

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
        x, y = batch
        x, y = self._prepare_data(x, y)

        logits = self(x)

        y = y[:, -1, :]

        if self._task == "classification":
            preds = torch.sigmoid(logits)
        else:
            preds = logits

        return preds, y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
            fused=True,
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
