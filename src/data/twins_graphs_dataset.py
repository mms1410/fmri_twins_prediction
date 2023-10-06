"""Defines Twin Dataset of two graphs with labels (twin y/n)."""

import os
import re
from itertools import combinations as itcomb
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

def create_participants_pairs_df(df_metadata, factor_non_twins, root):
    df_metadata = df_metadata.loc[:, [PARTICIPANT_ID, FAMILY_ID]]
    """Create pairs of participants. All pairs of twins are sampled. 
    Furthermore, factor_non_twins times the number of twins of pairs that are not twins are sampled.  """
    twin_pairs = df_metadata.groupby('family_id')['participant_id'].apply(list).tolist()
    num_non_twin_pairs = len(twin_pairs) * factor_non_twins
    non_twin_df = df_metadata.sample(n=num_non_twin_pairs * 2, random_state=42,  replace=True
                                     ).reset_index(drop=True)
    non_twin_df = non_twin_df.sample(frac=1, random_state=42)#shuffle df 
    non_twin_pairs = non_twin_df.groupby(non_twin_df.index // 2)['participant_id'].apply(list).tolist()
    all_pairs = twin_pairs + non_twin_pairs
    all_pairs_data = []
    for pair in all_pairs:
        try:
            participant_id, twin_id = pair
        except:            
            #some twins do not have a twin in the data
            continue
        family_id = df_metadata[df_metadata['participant_id'] == participant_id]['family_id'].values[0]
        are_twins = 1 if twin_id is not None and family_id == df_metadata[df_metadata['participant_id'] == twin_id]['family_id'].values[0] else 0
        all_pairs_data.append({'participant_id': participant_id, 'participants_twin_id': twin_id, 'family_id': family_id, 'are_twins': are_twins})

    
    #some files do not exist, remove those rows:
        # Assuming available_files contains the list of available filenames
    final_df = pd.DataFrame(all_pairs_data)

    logger = get_logger()
    logger.info(f"number of pairs before filter for valid files: {final_df.shape}")
    available_files = os.listdir(root)#["func_sub-0758_ses-01.pt", "func_sub-1104_ses-01.pt", ...]

    # Filter rows where either participant_id or participants_twin_id don't have matching filenames
    final_df = final_df[final_df.apply(lambda row: (f"func_{row['participant_id']}_ses-01.pt" in available_files) and 
                            (f"func_{row['participants_twin_id']}_ses-01.pt" in available_files), axis=1)]

    logger.info(f"number of pairs after filter for valid files: {final_df.shape}")
    
    
    # Shuffle the rows randomly
    shuffled_df = final_df.sample(frac=1, random_state=42) 

    # Now 'final_df' is a DataFrame with 'participant_id', 'family_id', 'participants_twin_id', and 'are_twins' columns
    return shuffled_df

class TwinsConnectomeDataset(Dataset):
    """Dataset of connectivity matrices."""

    def __init__(self, root: Union[str, Path] = Path("data", "processed"), 
                 meta_file: Union[str, Path] = Path("data", "raw", "ds004169", "participants.tsv")):
        """Initilaize Connectome Database.

        Args:
            root: string for path to folder with connectivites in pt format.
            meta_file: string for path to tabular with metadata.
        """
        super().__init__(root)
        self.df_metadata : pd.DataFrame = pd.read_table(meta_file)
        self.df_metadata : pd.DataFrame = create_participants_pairs_df(
            self.df_metadata, 1, root = root
        )
        #self.participant_ids : pd.Series = self.df_metadata[PARTICIPANT_ID]#os.listdir(root)

    def len(self):
        """Get length of Dataset.

        Returns:
            integer for elements in root.
        """
        return len(self.df_metadata.index)#len(os.listdir(self.root))

    def get(self, pair_idx):
        """Get Connectome Graph of patient.

        Args:
            idx: interger for indexed element in root.

        Returns:
            pyGeom Data class.
        """
        #id = extract_subid(filename)
        twins_metadata_df : pd.Series = self.df_metadata.iloc[pair_idx, :]
        family_id : str = twins_metadata_df[FAMILY_ID]
        participant_id : str = twins_metadata_df[PARTICIPANT_ID]
        participants_twin_id : str = twins_metadata_df[PARTICIPANT_TWIN_ID]
        
        #family_id = self.metadata[self.metadata["participant_id"] == id]
        #family_id = family_id["family_id"].iloc[0]
        participant_pt_filename = f"func_{participant_id}_ses-01.pt" # 
        try:
            graph_data = self.get_graph_of_participant(participant_pt_filename, participant_id)  
        except: 
            participant_pt_filename = f"func_{participant_id}_ses-02.pt" # sometimes only second session, e.g. with 1267
            graph_data = self.get_graph_of_participant(participant_pt_filename, participant_id)  
        



        # participants_twin_metadata : pd.Series = self.df_metadata[
        #     (self.df_metadata[FAMILY_ID] == family_id) &
        #                           (self.df_metadata[PARTICIPANT_ID] != participant_id)                                  
        #                           ]
        # participants_twin_id : int = participants_twin_metadata[PARTICIPANT_ID].iloc[0]

        participants_twin_pt_filename =  f"func_{participants_twin_id}_ses-01.pt" # 

        graph_twin_data = self.get_graph_of_participant(participants_twin_pt_filename, participants_twin_id)
        label : int = twins_metadata_df[ARE_TWINS]
        label = torch.tensor(label, dtype=torch.int64)
#        print(graph_data, graph_twin_data, label)

        return (graph_data, graph_twin_data), label

    def get_graph_of_participant(self, filename, participant_id) -> Data: 
        adjacency = torch.load(Path(self.root, filename))
        #node_features = torch.sum(adjacency, 1)
        node_features = torch.sum(adjacency, 1).unsqueeze(1)
        edge_idx, edge_attr = torch_geometric.utils.dense_to_sparse(adjacency)
        graph_data = Data(x=node_features, edge_index=edge_idx,  # noqa: BLK 100
                          edge_attr=edge_attr, y=participant_id)                          

        return graph_data
    
    @property
    def num_node_features(self):
        data, _ = self.get(0)  # Gets the first data point
        return data[0].num_node_features



if __name__ == "__main__":
    # project_dir = Path(__file__).resolve().parents[2]
    # connmats = Path(project_dir, "data", "processed")
    # metadata = Path(project_dir, "data", "raw", "ds004169", "participants.tsv")
    # metadata = pd.read_csv(metadata, delimiter="\t")
    twins_dataset = TwinsConnectomeDataset()
    print(twins_dataset.df_metadata)
    print(twins_dataset[0])
    print(twins_dataset[1])
    print(twins_dataset[2])
    print(twins_dataset[3])
#    print(twins_dataset[3][0][0].shape)
    
    

