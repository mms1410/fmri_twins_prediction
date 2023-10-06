import logging

import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ProgressBar
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.data.twins_graphs_dataset import TwinsConnectomeDataset
from src.models.model import TwinGNN
from src.util import get_logger

#pl._logger.setLevel(logging.DEBUG)

class LitProgressBar(ProgressBar):
    def __init__(self):
        super().__init__()
        self.init_progress_bar()

    def init_progress_bar(self):
        self.main_progress_bar = tqdm(desc='Training')
        return self.main_progress_bar

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, *args)
        if trainer.sanity_checking:
            return

        # Print the current epoch number
        print(f"Epoch: {trainer.current_epoch}")

        # Print the current batch index
        print(f"Batch: {batch_idx}")

        # Print the size of the batch
        print(f"Batch size: {len(batch)}")

        # Print the output of the model for this batch
        print(f"Model output: {outputs}")

        # If you have logged metrics in your `training_step`, you can access them like this:
        train_loss = pl_module.trainer.callback_metrics.get('train_loss')
        print(f"train_loss:{train_loss}")
        eval_loss = pl_module.trainer.callback_metrics.get('eval_loss')

        print(f"eval_loss:{eval_loss}")

        if train_loss:
            self.main_progress_bar.set_postfix(train_loss=train_loss.item(), refresh=False)

        accuracy = pl_module.trainer.callback_metrics.get('accuracy')
        precision = pl_module.trainer.callback_metrics.get('precision')
        recall = pl_module.trainer.callback_metrics.get('recall')

        print(f"accuracy:{accuracy}")
        print(f"precision:{precision}")
        print(f"recall:{recall}")


# 1. Start MLflow tracking
mlflow.start_run()

twins_dataset = TwinsConnectomeDataset()

# 2. Split your dataset into training and validation sets
train_size = int(0.98 * len(twins_dataset))
val_size = len(twins_dataset) - train_size
logger = get_logger()
logger.info(f"val_size: {val_size}")

train_dataset, val_dataset = random_split(twins_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

model = TwinGNN(in_channels=1, hidden_channels=64)
model.float()

class MLFlowLoggingCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        for key, value in logs.items():
            if key in ['train_loss_epoch', 'val_loss_epoch']:
                mlflow.log_metric(key, value)


default_progress_bar = pl.callbacks.ProgressBar()

trainer = pl.Trainer(
    max_epochs=1,
    accelerator='auto',
    devices="auto",
    callbacks=[MLFlowLoggingCallback(), LitProgressBar()
               #, default_progress_bar
               ],
    
    val_check_interval=0.0000001,
    #val_check_interval=1
)


# Provide both training and validation dataloaders to trainer.fit()
trainer.fit(model, train_loader, val_loader)

# End MLflow tracking
mlflow.end_run()
