"""Functions used to load a trained model."""
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from src.models.model import TwinGNN


def get_model_folder(log_dir: Union[Path, str], project_name: Union[Path, str], experiment_name: Optional[str] = ""):  # noqa E501
    """Get folder with logged model results and checkpoint.

    Args:
        log_dir: path to logs.
        project_name: name of project's name.
        experiment_name: name indicating which version to use.
                         If empty latest version is used.

    Returns:
        Path object to model specific log files.
    """
    model_log = Path(log_dir, project_name)
    # if experiment name not specified get latest version
    if experiment_name == "":
        subdirectories = [d for d in model_log.iterdir(
        ) if d.is_dir() and d.name.startswith("version_")]
        sorted_subdirectories = sorted(
            subdirectories, key=lambda d: int(d.name.split("_")[1]), reverse=True)  # noqa E501
        model_folder = sorted_subdirectories[0].name
        model_folder = Path(log_dir, project_name,
                            model_folder)  # type: ignore
    else:
        model_folder = Path(log_dir, project_name,
                            experiment_name)  # type: ignore
    return model_folder


def get_model(experiment_folder: Path, checkpoint_name: Optional[str] = ""):  # noqa E501
    """Get the actual model from given folder.

    Args:
        experiment_folder: path to logged data and checkpoints.
        checkpoint_name: name of checkpoint to load, otw latest.

    Returns:
        Path to model.
    """
    checkpoint = Path(experiment_folder, "checkpoints")
    if checkpoint_name == "":
        checkpoint = next(checkpoint.iterdir())  # get first elem
    else:
        checkpoint = Path(checkpoint, checkpoint_name)  # type: ignore
    # get model configuration
    conf = OmegaConf.load(Path(experiment_folder, "hparams.yaml"))
    # init model
    model = TwinGNN(in_channels=1, hidden_channels=conf.hidden_channels)
    # load model
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])  # type: ignore
    model.eval()
    return model


def write_loss_plot(file: Path, destination_file: Path, loss: str = "val_loss") -> None:  # noqa E501
    """Write plot of validation error.

    Args:
        file: Path of csv file with necessary data.
        destination_file: Complete Path of jpg file that will be created.
        loss: string for value of y-axis. Must be in csv file.
    """
    metrics = pd.read_csv(file)
    idx = metrics.groupby('epoch')['step'].idxmax()
    metrics = metrics.loc[idx]
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['epoch'], metrics['val_loss'], marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Error')
    plt.title('Validation Error vs. Epoch')
    plt.grid(True)
    plt.savefig(destination_file, format='jpg')


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]
    folder = get_model_folder(log_dir=Path(
        project_dir, "logs"), project_name="twins_connectome_project")
    write_loss_plot(Path(folder, "metrics.csv"), Path(
        project_dir, "assets", "validation_loss.jpg"))
