"""saves the downloaded data under data / raw.

installation of datalad:
https://docs.datalad.org/en/latest/generated/man/datalad-clone.html
e.g. to install in ubuntu:
sudo apt-get install datalad

"""
import os
import shutil

import datalad.api as dl

id = "ds004169"

dataset_url = f"https://github.com/OpenNeuroDatasets/{id}.git"

dataset = dl.clone(source=dataset_url)
dl.get(dataset.path)

destination_dir = os.path.join("data", "raw", id)

if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Move the folder to the destination
shutil.move(id, destination_dir)

# todo: logging
