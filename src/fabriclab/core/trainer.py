import torch
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
        self.dataloader = None
        self.optimizer = None

    def fit(self, model, dataset, optimizer) -> None:
        self.model, self.optimizer = self.fabric.setup(model, optimizer)
        self.dataloader = self.fabric.setup_dataloaders(dataset)

        self.fabric.launch()

        self.model.train()
        for epoch in range(self.max_epochs):
            for batch in self.dataloader:
                input, target = batch
                self.optimizer.zero_grad()
                output = self.model(input, target)
                loss = torch.nn.functional.nll_loss(output, target.view(1))
                self.fabric.backward(loss)
                self.optimizer.step()
