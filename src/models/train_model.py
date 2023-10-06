"""Train model with MRI data."""

import pytorch_lightning as pl
from torch_geometric.data import DataLoader

from src.data.twins_graphs_dataset import TwinsConnectomeDataset
from src.models.model import TwinGNN

twins_dataset = TwinsConnectomeDataset()

model = TwinGNN(in_channels=1, hidden_channels=64)
model.float() 
trainer = pl.Trainer(max_epochs=1, accelerator='auto', devices="auto")
dataloader = DataLoader(twins_dataset, batch_size=2, shuffle=True)
trainer.fit(model, dataloader)


