"""Module to provide utilities for other modules."""

import os
import re
from itertools import combinations as itcomb
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


def get_positive_pairs(metadata_df: pd.DataFrame):
    """Get list of positive (twins) pairs.

    This function creates a list of tuples. Each tuple consists
    of a subject_id and another subject_id with same family_id (e.g. twins).

    Args:
        metadata_df: pandas dataframe with col 'participant_id' and 'family_id'

    Returns:
        list of tuples with subject id pairs.
    """
    pairs = metadata_df.groupby('family_id')['participant_id'].apply(
        lambda x: list(itcomb(x, 2)))
    # Flatten the list of pairs
    pos_pairs = [pair for pairs_list in pairs.tolist()
                 for pair in pairs_list]
    return pos_pairs


def get_negative_pairs(metadata_df: pd.DataFrame):
    """
    Get list of negative (no twins) pairs.

    Args:
        metadata_df: pandas dataframe with col 'participant_id' and 'family_id'

    Returns:
        list of tuples with subject id pairs.
    """
    neg_pairs = []
    for pat_id, fam_id in zip(metadata_df["participant_id"], metadata_df["family_id"]):  # noqa: E501

        filtered_df = metadata_df[metadata_df["family_id"] != fam_id]
        # random pick
        random_pick = filtered_df.sample(1)
        pick_pat_id = random_pick["participant_id"].iloc[0]
        pair = (pat_id, pick_pat_id)
        neg_pairs.append(pair)
    return neg_pairs


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]
    connmats = Path(project_dir, "data", "processed")
    metadata = Path(project_dir, "data", "raw", "ds004169", "participants.tsv")
    metadata = pd.read_csv(metadata, delimiter="\t")
    neg_pairs = get_negative_pairs(metadata)
    pos_pairs = get_positive_pairs(metadata)
