from lightning_fabric import Fabric


class LabTrainer:
    def __init__(
        self,
        devices="auto",
        accelerator="auto",
        strategy="auto",
        num_nodes=1,
        precision="32-true",
        max_epochs=2,
        logger=None,
    ) -> None:
        self.fabric = Fabric(
            devices=devices,
            accelerator=accelerator,
            strategy=strategy,
            num_nodes=num_nodes,
            precision=precision,
            loggers=logger,
        )
        self.max_epochs = max_epochs
        self.model = None
        self.datamodule = None

        if devices > 1:
            self.fabric.launch()

    def fit(self, model, datamodule) -> None:
        self.model = model
        self.datamodule = datamodule

        self.datamodule.prepare_data()
        self.datamodule.setup()

        for i in range(self.max_epochs):
            ...
