import logging

from typing_extensions import override

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback

logger = logging.getLogger(__name__)


class DGLGraphToDevice(Callback):
    def __init__(self) -> None:
        pass

    @override
    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        logger.info("on fit start called")

    @override
    def setup(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
    ) -> None:
        logger.info(f"setup called on stage: {stage}")
        pl_module.dglTo(pl_module.device)
