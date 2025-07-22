import os

import pandas as pd
import torch
from torch_geometric.data import Data


def create_graph_data_object(data_dir="data/raw/elliptic_bitcoin_dataset"):
    """
    Read raw CSVs and convert them into a PyG Data object.
    Save the processed object for later use.
    """
    print("Loading raw data...")
    # Load features and classes
    features_df = pd.read_csv(
        os.path.join(data_dir, "elliptic_txs_features.csv"), header=None
    )
    classes_df = pd.read_csv(os.path.join(data_dir, "elliptic_txs_classes.csv"))

    # Load edge list
    edgelist_df = pd.read_csv(os.path.join(data_dir, "elliptic_txs_edgelist.csv"))

    # Rename feature columns
    # First column is txId, second is timestep, rest are features
    feature_names = (
        ["txId", "timestep"]
        + [f"feature_{i}" for i in range(93)]
        + [f"agg_feature_{i}" for i in range(72)]
    )
    features_df.columns = feature_names

    # Merge features with classes
    # Use a left merge to keep all transactions, even those without a class label
    full_df = pd.merge(features_df, classes_df, on="txId", how="left")
    full_df["class"] = full_df["class"].fillna("unknown")  # Fill NaNs

    # === Preprocessing and Node Mapping ===
    # PyG requires node indices to be continuous from 0 to N-1
    # We create a mapping from the original txId to a new integer index
    tx_ids = full_df["txId"].values
    tx_id_map = {tx_id: i for i, tx_id in enumerate(tx_ids)}

    print("Processing node features and labels...")
    # Select only the numeric features for the model
    # We exclude txId and timestep for now
    node_features = full_df.drop(columns=["txId", "timestep", "class"]).values

    # Convert to PyTorch tensor
    x = torch.tensor(node_features, dtype=torch.float)

    # Process labels: '1' (illicit) -> 1, '2' (licit) -> 0, 'unknown' -> -1
    y = torch.tensor(
        full_df["class"].map({"1": 1, "2": 0, "unknown": -1}).values, dtype=torch.long
    )

    # === Edge Index Construction ===
    print("Processing edge index...")
    # Map the txIds in the edgelist to our new integer indices
    source_nodes = edgelist_df["txId1"].map(tx_id_map).values
    target_nodes = edgelist_df["txId2"].map(tx_id_map).values

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

    # === Create masks for training, validation, and testing ===
    # We only train on the labeled nodes
    # The original paper uses specific time steps for train/test split.
    # For simplicity, we'll create masks based on labeled data.
    train_mask = torch.tensor(y != -1, dtype=torch.bool)
    # We'll split the labeled data in the training script later.

    # === Assemble the Data object ===
    graph_data = Data(x=x, edge_index=edge_index, y=y)
    graph_data.train_mask = train_mask
    graph_data.tx_id_map = tx_id_map

    print("Graph construction complete.")
    print(graph_data)

    # Save the processed data object
    processed_dir = os.path.join(data_dir, "..", "..", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    torch.save(graph_data, os.path.join(processed_dir, "graph_data.pt"))
    print(f"Graph data object saved to {os.path.join(processed_dir, 'graph_data.pt')}")


if __name__ == "__main__":
    create_graph_data_object()
