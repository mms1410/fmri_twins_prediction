"""Module to provide utilities for other modules."""

import os
import re
from pathlib import Path
from typing import Union

import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset


class ConnectomeDatabase(Dataset):
    """Database of connectivity matrices."""

    def __init__(self, root: Union[str, Path], meta_file: Union[str, Path]):
        """Initilaize Connectome Database.

        Args:
            root: string for path to folder with connectivites in pt format.
            meta_file: string for path to tabular with metadata.
        """
        super().__init__(root)
        self.metadata = pd.read_table(meta_file)
        self.subjects = os.listdir(root)

    def len(self):
        """Get length of Dataset.

        Returns:
            integer for elements in root.
        """
        return len(os.listdir(self.root))

    def get(self, idx):
        """Get Connectome Graph of patient.

        Args:
            idx: interger for indexed element in root.

        Returns:
            pyGeom Data class.
        """
        filename = self.subjects[idx]
        id = extract_subid(filename)
        family_id = self.metadata[self.metadata["participant_id"] == id]
        family_id = family_id["family_id"].iloc[0]
        adjacency = torch.load(Path(self.root, filename))
        node_features = torch.sum(adjacency, 1)
        edge_idx, edge_attr = torch_geometric.utils.dense_to_sparse(adjacency)
        graph_data = Data(x=node_features, edge_index=edge_idx,  # noqa: BLK 100
                          edge_attr=edge_attr, y=family_id)  # noqa: BLK 100
        return graph_data


def extract_subid(filename: str) -> str:
    """Extract the subjects id from filename.

    Args:
        filename: string of filename.

    Returns:
        string of subject id.

    Raises:
        NoMatchFoundError: If no match found.
    """
    pattern = r'sub-\d+'
    match = re.search(pattern, filename)
    if match:
        return match.group(0)
    else:
        raise NoMatchFoundError(pattern)


class NoMatchFoundError(Exception):
    """Error Class for regex."""

    def __init__(self, pattern):
        """Init error class.

        Args:
            pattern: string with regular expression.
        """
        self.pattern = pattern
        super().__init__(f"No match found for pattern: {pattern}")


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]
    connmats = Path(project_dir, "data", "processed")
    metadata = Path(project_dir, "data", "raw", "ds004169", "participants.tsv")
    connbase = ConnectomeDatabase(root=connmats, meta_file=metadata)
    sample = connbase[0]
