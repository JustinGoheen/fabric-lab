import torch
from torch.utils.data import DataLoader
from lightning_fabric import Fabric
from fabriclab import LabDataset, LabModule


fabric = Fabric(
    accelerator="auto",
    strategy="auto",
    devices="auto",
    num_nodes=1,
    precision="32-true",
)
fabric.launch()

# prepare the dataset
dataset = LabDataset()
dataloader = DataLoader(dataset)

# prepare the model
model = LabModule(dataset.vocab_size)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

model, optimizer = fabric.setup(model, optimizer)
dataloader = fabric.setup_dataloaders(dataloader)

model.train()
for epoch in range(20):
    for idx, batch in enumerate(dataloader):
        input, target = batch
        optimizer.zero_grad()
        output = model(input, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        fabric.backward(loss)
        fabric.print(f"EPOCH {epoch} BATCH {idx} LOSS: {loss}")
        optimizer.step()
