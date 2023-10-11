"""Module to train the twin classification nn."""
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.data.twins_graphs_dataset import TwinsConnectomeDataset
from src.models.model import TwinGNN
from src.util import get_logger

project_dir = Path(__file__).resolve().parents[2]
config = OmegaConf.load("configs/data_training.yaml")
conf_hidden_channels = config.hidden_channels
conf_epochs = config.epochs
conf_batch_size = config.batch_size


class LitProgressBar(ProgressBar):
    """Class to create own progress bar.

    Args:
        ProgressBar: super class
    """

    def __init__(self):
        """Init progress bar."""
        super().__init__()
        self._init_progress_bar()

    def _init_progress_bar(self):
        self.main_progress_bar = tqdm(desc="Training")
        return self.main_progress_bar

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args):
        """Call back method on training end.

        Args:
            trainer : trainer
            pl_module : pl module
            outputs : outputs
            batch : batch
            batch_idx : batch_idx
            args : further arguments for backward compatibility
        """
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, *args)
        if trainer.sanity_checking:
            return

        # Print the current epoch number
        logger.info(f"Epoch: {trainer.current_epoch}")

        # Print the current batch index
        logger.info(f"Batch: {batch_idx}")

        # Print the size of the batch
        logger.info(f"Batch size: {len(batch)}")

        # Print the output of the model for this batch
        logger.info(f"Model output: {outputs}")

        # If you have logged metrics in your `training_step`
        # you can access them like this:
        train_loss = pl_module.trainer.callback_metrics.get("train_loss")
        logger.info(f"train_loss:{train_loss}")
        eval_loss = pl_module.trainer.callback_metrics.get("eval_loss")

        logger.info(f"eval_loss:{eval_loss}")

        if train_loss:
            self.main_progress_bar.set_postfix(  # noqa NLK 100
                train_loss=train_loss.item(), refresh=False)  # noqa NLK 100

        accuracy = pl_module.trainer.callback_metrics.get("accuracy")
        precision = pl_module.trainer.callback_metrics.get("precision")
        recall = pl_module.trainer.callback_metrics.get("recall")

        logger.info(f"accuracy:{accuracy}")
        logger.info(f"precision:{precision}")
        logger.info(f"recall:{recall}")


twins_dataset = TwinsConnectomeDataset()

# 2. Split your dataset into training and validation sets
train_size = int(0.98 * len(twins_dataset))
val_size = len(twins_dataset) - train_size
logger = get_logger()
logger.info(f"val_size: {val_size}")

train_dataset, val_dataset = random_split(
    twins_dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset, batch_size=conf_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=conf_batch_size, shuffle=False)

model = TwinGNN(in_channels=1, hidden_channels=conf_hidden_channels)
model.float()


# set csv logger to track training metrics
logger_csv = CSVLogger("logs",
                       name="twins_connectome_project")
logger_csv.log_hyperparams(params=config)

# set callback to save top k models
checkpoint_callback = ModelCheckpoint(
    dirpath=Path(project_dir, "models",
                 "checkpoints"),
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

default_progress_bar = pl.callbacks.ProgressBar()

trainer = pl.Trainer(
    max_epochs=conf_epochs,
    accelerator="auto",
    devices="auto",
    callbacks=[LitProgressBar(), checkpoint_callback],
    val_check_interval=0.0000001,
    # val_check_interval=1
)

trainer.fit(model, train_loader, val_loader)
