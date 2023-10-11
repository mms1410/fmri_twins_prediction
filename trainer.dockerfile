FROM python:3.9-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/* 

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY README.md README.md
COPY src/ src/
COPY configs/ configs/
COPY data/processed/ data/processed/
COPY data/raw/ds004169/participants.tsv data/raw/ds004169/participants.tsv

COPY README.md README.md

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install .

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
