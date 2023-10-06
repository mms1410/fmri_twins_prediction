import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv


class TwinGNN(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels):
        super(TwinGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(2 * hidden_channels, 1)

    def forward_one(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


    def forward(self, data):
        # Assuming each batch has the shape [batch_size, num_nodes, features]
        graph_data, graph_twin_data = data

        batch_size = graph_data.num_graphs
        out_list = []

        for idx in range(batch_size):
            single_graph_data = graph_data[idx]
            single_graph_twin_data = graph_twin_data[idx]

            single_graph_data.x = single_graph_data.x.float()
            single_graph_twin_data.x = single_graph_twin_data.x.float()

            out1 = self.forward_one(single_graph_data.x, single_graph_data.edge_index)
            out2 = self.forward_one(single_graph_twin_data.x, single_graph_twin_data.edge_index)
            
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
        (data, labels) = batch
        out = self(data)
        
        # Reshape labels to match the output shape
        labels = labels.view(out.shape).float()
        criterion = torch.nn.BCELoss()
        loss = criterion(out, labels)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)