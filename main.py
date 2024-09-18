#!/usr/bin/env python3

from seasfire.data import SeasFireDataModule
from seasfire.firecastnet_lit import FireCastNetLit
from seasfire.gru_lit import GRULit
from seasfire.conv_gru_lit import ConvGRULit
from seasfire.conv_lstm_lit import ConvLSTMLit
from seasfire.utae_lit import UTAELit
from seasfire.cli import SeasfireLightningCLI
import logging

logger = logging.getLogger(__name__)


class GRU(GRULit):
    def configure_optimizers(self):
        logger.info(f"⚡ Using GRU ⚡")
        return super().configure_optimizers()


class ConvGRU(ConvGRULit):
    def configure_optimizers(self):
        logger.info(f"⚡ Using ConvGRU ⚡")
        return super().configure_optimizers()


class ConvLSTM(ConvLSTMLit):
    def configure_optimizers(self):
        logger.info(f"⚡ Using ConvLSTM ⚡")
        return super().configure_optimizers()


class UTAE(UTAELit):
    def configure_optimizers(self):
        logger.info(f"⚡ Using UTAE ⚡")
        return super().configure_optimizers()


class FireCastNet(FireCastNetLit):
    def configure_optimizers(self):
        logger.info(f"⚡ Using FireCastNet ⚡")
        return super().configure_optimizers()


def main():
    level = logging.INFO
    logging.basicConfig(level=level)

    cli = SeasfireLightningCLI(
        datamodule_class=SeasFireDataModule,
    )


if __name__ == "__main__":
    main()
