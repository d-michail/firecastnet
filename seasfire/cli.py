from lightning.pytorch.cli import LightningCLI


class SeasfireLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.max_epochs", "model.init_args.max_epochs")

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.task", "model.init_args.task")
