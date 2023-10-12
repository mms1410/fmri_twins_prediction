"""Module for inference for twins."""

from pathlib import Path

import click  # Import the click library

from src.data.twins_graphs_dataset import TwinsConnectomeDataset
from src.models.utils import get_model, get_model_folder
from src.util import get_logger


@click.command()
@click.option("--pair-number", default=42, help="Number of pair to predict")
def main(pair_number):
    """Predict whether the pair are twins.

    Args:
        pair_number: number of the twin pair
    """
    project_dir = Path(__file__).resolve().parents[2]
    folder = get_model_folder(
        log_dir=Path(project_dir, "logs"), project_name="twins_connectome_project"
    )
    model = get_model(folder)

    twins_dataset = TwinsConnectomeDataset()
    graph1, graph2 = twins_dataset[pair_number][0]
    graph1 = [graph1]
    graph2 = [graph2]
    input_graphs = (graph1, graph2)
    output = model(input_graphs)
    logger = get_logger()
    logger.info(f"the probability whether they are twins is {output}")
    logger.info(f"the prediction whether they are twins is {int(output)}")


if __name__ == "__main__":
    main()
