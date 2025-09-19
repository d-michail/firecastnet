#!/usr/bin/env python3

from seasfire.data import SeasFireDataModule
from seasfire.cli import SeasfireLightningCLI
import logging

logger = logging.getLogger(__name__)

def main():
    level = logging.INFO
    logging.basicConfig(level=level)

    cli = SeasfireLightningCLI(
        datamodule_class=SeasFireDataModule,
        save_config_callback=None
    )


if __name__ == "__main__":
    main()
