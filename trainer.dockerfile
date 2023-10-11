# Base image
FROM python:3.9-slim

# install python and create venv file if necessary
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/* 

# copy application
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
# do not include data

# set working dir and install dependencies
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# set entrypoint which will run when image executed (redirect print to terminal with u)
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]

# files created during training are located within the container
# to copy files between a completed run to local machine use 'docker cp', e.g.:
# docker cp {container_name}:{dir_path}/{file_name} {local_dir_path}/{local_file_name}