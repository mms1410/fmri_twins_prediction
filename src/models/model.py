"""modul to hold the twin classification network."""
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as nn_func
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

config = OmegaConf.load("configs/data_training.yaml")


class TwinGNN(pl.LightningModule):
    """Class for nns that can process scans of two twins."""

    def __init__(self, in_channels, hidden_channels):
        """Init network.

        Args:
            in_channels (_type_): no of in channels
            hidden_channels (_type_): no of hidden channels
        """
        super(TwinGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(2 * hidden_channels, 1)

    def _forward_one(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = nn_func.relu(x)
        x = nn_func.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def forward(self, data: tuple):
        """Forward function of the nn.

        Args:
            data (tuple): pair of participant's and twinś garaph dat

        Returns:
            : _description_
        """
        # Assuming each batch has the shape [batch_size, num_nodes, features]
        graph_data: Data = data[0]
        graph_twin_data: Data = data[1]

        batch_size = graph_data.num_graphs
        out_list = []

        for idx in range(batch_size):
            single_graph_data = graph_data[idx]
            single_graph_twin_data = graph_twin_data[idx]

            single_graph_data.x = single_graph_data.x.float()
            single_graph_twin_data.x = single_graph_twin_data.x.float()

            out1 = self._forward_one(  # noqa NLK 100
                single_graph_data.x, single_graph_data.edge_index)  # noqa NLK 100
            out2 = self._forward_one(
                single_graph_twin_data.x, single_graph_twin_data.edge_index)

            # Aggregate node representations
            out1 = out1.mean(dim=0)
            out2 = out2.mean(dim=0)

            # Combine representations
            combined = torch.cat([out1, out2], dim=0)
            out = self.fc(combined).view(1, -1)
            out_list.append(out)

        final_output = torch.cat(out_list, dim=0)
        return torch.sigmoid(final_output)

    def training_step(self, batch, batch_idx):
        """Make training step.

        Args:
            batch (_type_): batch
            batch_idx (_type_): batch_id

        Returns:
            _type_: output of training step
        """
        (data, labels) = batch
        out = self(data)

        # Reshape labels to match the output shape
        labels = labels.view(out.shape).float()
        criterion = torch.nn.BCELoss()
        loss = criterion(out, labels)
        self.log("train_loss", loss)

        return {"loss": loss, "progress_bar": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        """Make validation step.

        Args:
            batch (_type_): batch
            batch_idx (_type_): id of batch
        """
        (data, labels) = batch
        out = self(data)
        labels = labels.view(out.shape).float()
        criterion = torch.nn.BCELoss()
        val_loss = criterion(out, labels)
        self.log("val_loss", val_loss, batch_size=len(batch))

        # Convert the output probabilities to binary values (0 or 1)
        out_binary = (out > 0.5).float()

        # Calculate accuracy, precision and recall
        accuracy = accuracy_score(labels.cpu(), out_binary.cpu())
        precision = precision_score(
            labels.cpu(), out_binary.cpu(), zero_division=np.nan)
        recall = recall_score(
            labels.cpu(), out_binary.cpu(), zero_division=np.nan)

        # Log the metrics
        self.log("accuracy", accuracy, batch_size=len(batch))
        self.log("precision", precision, batch_size=len(batch))
        self.log("recall", recall, batch_size=len(batch))

    def configure_optimizers(self):
        """Configure optimizer.

        Returns:
            _type_: optimizer
        """
        partial_optimizer = instantiate(config.optimizer)
        optimizer = partial_optimizer(self.parameters())
        return optimizer
