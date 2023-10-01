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


def download_dataset(dataset_accession_number: str = "ds004169"):
    """Download dataset from open neuro datasets.

    Args:
        dataset_accession_number (str): neuro number.
    """
    url_part1 = "https://github.com/OpenNeuroDatasets/"
    urlpart2 = f"{dataset_accession_number}.git"
    dataset_url = url_part1 + urlpart2

    dataset = dl.clone(source=dataset_url)
    dl.get(dataset.path)

    destination_dir = os.path.join("data", "raw", dataset_accession_number)

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Move the folder to the destination:
    shutil.move(dataset_accession_number, destination_dir)
    logger = logging.getLogger(__name__)
    logger.info(
        f"raw fmri data {dataset_accession_number} downloaded and " f"saved to {destination_dir}"
    )


if __name__ == "__main__":
    download_dataset()
