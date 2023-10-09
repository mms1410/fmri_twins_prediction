"""Defines Twin Dataset of two graphs with labels (twin y/n)."""
import logging
import os
from pathlib import Path
from typing import Union

import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset

from src.util import get_logger

FAMILY_ID = "family_id"
PARTICIPANT_ID = "participant_id"
PARTICIPANT_TWIN_ID = "participants_twin_id"
ARE_TWINS = "are_twins"


def create_participants_pairs_df(
    df_metadata: pd.DataFrame, factor_non_twins: int, root: Union[str, Path]
):
    """Create pairs of participants. All pairs of twins are sampled.

    Furthermore, factor_non_twins times the number of twins of pairs
     that are not twins are sampled.

    Args:
        df_metadata: metadata
        factor_non_twins (int): _factor non twins
        root : root

    Returns:
        pd.DataFrame: df with participant pairs_
    """
    df_metadata = df_metadata.loc[:, [PARTICIPANT_ID, FAMILY_ID]]

    twin_pairs = df_metadata.groupby("family_id")["participant_id"].apply(list).tolist()
    num_non_twin_pairs = len(twin_pairs) * factor_non_twins
    non_twin_df = df_metadata.sample(
        n=num_non_twin_pairs * 2, random_state=42, replace=True
    ).reset_index(drop=True)
    non_twin_df = non_twin_df.sample(frac=1, random_state=42)  # shuffle df
    non_twin_pairs = (
        non_twin_df.groupby(non_twin_df.index // 2)["participant_id"].apply(list).tolist()
    )
    all_pairs = twin_pairs + non_twin_pairs
    all_pairs_data = []
    for pair in all_pairs:
        try:
            participant_id, twin_id = pair
        except Exception as e:
            # some twins do not have a twin in the data
            (f"Error unpacking pair: {e}")
            continue
        family_id = df_metadata[df_metadata["participant_id"] == participant_id][
            "family_id"
        ].values[0]
        are_twins = (
            1
            if twin_id is not None
            and family_id
            == df_metadata[df_metadata["participant_id"] == twin_id]["family_id"].values[0]
            else 0
        )
        all_pairs_data.append(
            {
                "participant_id": participant_id,
                "participants_twin_id": twin_id,
                "family_id": family_id,
                "are_twins": are_twins,
            }
        )

    # some files do not exist, remove those rows:
    # Assuming available_files contains the list of available filenames
    final_df = pd.DataFrame(all_pairs_data)

    logger = get_logger()
    logger.info(f"number of pairs before filter for valid files: {final_df.shape}")
    available_files = os.listdir(
        root
    )  # ["func_sub-0758_ses-01.pt", "func_sub-1104_ses-01.pt", ...]

    # Filter rows where either participant_id or participants_twin_id don't have matching filenames
    final_df = final_df[
        final_df.apply(
            lambda row: (f"func_{row['participant_id']}_ses-01.pt" in available_files)
            and (f"func_{row['participants_twin_id']}_ses-01.pt" in available_files),
            axis=1,
        )
    ]

    logger.info(f"number of pairs after filter for valid files: {final_df.shape}")

    # Shuffle the rows randomly
    shuffled_df = final_df.sample(frac=1, random_state=42)

    # Now 'final_df' is a DataFrame with 'participant_id', 'family_id',
    # 'participants_twin_id', and 'are_twins' columns
    return shuffled_df


class TwinsConnectomeDataset(Dataset):
    """Dataset of connectivity matrices."""

    def __init__(
        self,
        root: Union[str, Path, None] = None,
        meta_file_path: Union[str, Path, None] = None,
    ):
        """Initialize Connectome Dataset.

        Args:
            root: string for path to folder with connectivites in pt format.
            meta_file_path: string for path to tabular with metadata.
        """
        if root is None:
            root = Path("data", "processed")
        if meta_file_path is None:
            meta_file_path = Path("data", "raw", "ds004169", "participants.tsv")
        super().__init__(root)
        self.df_metadata: pd.DataFrame = pd.read_table(meta_file_path)
        self.df_metadata = create_participants_pairs_df(self.df_metadata, 1, root=root)

    def len(self):
        """Get length of Dataset.

        Returns:
            integer for elements in root.
        """
        return len(self.df_metadata.index)

    def get(self, pair_idx: tuple):
        """Get Connectome Graph of patient.

        Args:
             pair_idx: interger for indexed element in metadata.

        Returns:
            pair of pyGeom Data class.
        """
        twins_metadata_df: pd.Series = self.df_metadata.iloc[pair_idx, :]
        participant_id: str = twins_metadata_df[PARTICIPANT_ID]
        participants_twin_id: str = twins_metadata_df[PARTICIPANT_TWIN_ID]

        participant_pt_filename = f"func_{participant_id}_ses-01.pt"
        graph_data = self.get_graph_of_participant(participant_pt_filename, participant_id)

        participants_twin_pt_filename = f"func_{participants_twin_id}_ses-01.pt"

        graph_twin_data = self.get_graph_of_participant(
            participants_twin_pt_filename, participants_twin_id
        )
        label: int = twins_metadata_df[ARE_TWINS]
        label = torch.tensor(label, dtype=torch.int64)

        return (graph_data, graph_twin_data), label

    def get_graph_of_participant(self, filename: str, participant_id) -> Data:
        """Get the pytorch geometric graph of the brainscan of a participant.

        Args:
            filename (str): filename
            participant_id (str): participant_id

        Returns:
            Data: graph
        """
        adjacency = torch.load(Path(self.root, filename))
        # node_features = torch.sum(adjacency, 1)
        node_features = torch.sum(adjacency, 1).unsqueeze(1)
        edge_idx, edge_attr = torch_geometric.utils.dense_to_sparse(adjacency)
        graph_data = Data(
            x=node_features,
            edge_index=edge_idx,  # noqa: BLK 100
            edge_attr=edge_attr,
            y=participant_id,
        )

        return graph_data

    @property
    def num_node_features(self):
        """Return number of node features.

        Returns:
            int: number of node features
        """
        data, _ = self.get(0)
        return data[0].num_node_features


if __name__ == "__main__":
    twins_dataset = TwinsConnectomeDataset()
    logging.info(twins_dataset.df_metadata)
    logging.info(twins_dataset[0])
    logging.info(twins_dataset[1])
    logging.info(twins_dataset[2])
    logging.info(twins_dataset[3])
