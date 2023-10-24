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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, model, dataloader, optimizer) -> None:
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer

        self.model = self.model.to(self.device)
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.dataloader = self.fabric.setup_dataloaders(self.dataloader)

        self.fabric.launch()

        self.model.train()
        for epoch in range(self.max_epochs):
            for batch in self.dataloader:
                input, target = batch
                input, target = input.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(input, target)
                loss = torch.nn.functional.nll_loss(output, target.view(1))
                loss.backward()
                self.fabric.backward(loss)
                self.optimizer.step()
