"""
saves the downloaded fmri data of twins under data/raw.

installation of datalad:
https://docs.datalad.org/en/latest/generated/man/datalad-clone.html
e.g. installation in ubuntu via:
sudo apt-get install datalad

"""
import logging
import os
import shutil

import datalad.api as dl


def download_dataset(
    dataset_access_number: str = "ds004169",
    destination_dir: str | None = None,
):
    """Download dataset from open neuro datasets.

    Args:
        dataset_access_number: neuro number (str, optional)
        destination_dir : path where the folders are stored (str | None, optional)
    """
    url_part1 = "https://github.com/OpenNeuroDatasets/"
    url_part2 = f"{dataset_access_number}.git"
    dataset_url = url_part1 + url_part2

    dataset = dl.clone(source=dataset_url)
    dl.get(dataset.path)

    if destination_dir is None:
        destination_dir = os.path.join("data", "raw", dataset_access_number)

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Move the folder to the destination:
    shutil.move(dataset_access_number, destination_dir)
    logger = logging.getLogger(__name__)
    logger.info(f"raw fmri data {dataset_access_number} downloaded and saved to {destination_dir}")


if __name__ == "__main__":
    download_dataset()