import numpy as np
import os
from scipy.spatial import cKDTree
import torch
from torch.utils.data import Dataset, DataLoader

def load_dataset(root_dir, alignment_strength):
    """
    Load the dataset from the given root directory.
    """
    data_components = ['positions', 'orientations', 'properties']
    data = {}

    for component in data_components:
        path = os.path.join(root_dir, 'preprocessed', f'{component}_{alignment_strength}.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file {path} does not exist.")

        data[component] = np.load(path)

    return data

def split_dataset(data):
    """
    Split the dataset into training and validation sets based on the x-axis.
    Always split at the midpoint.
    """
    x_coords = data['positions'][:, 0]
    x_mid = (x_coords.min() + x_coords.max()) / 2

    train_indices = np.where(x_coords <= x_mid)[0]
    val_indices = np.where(x_coords > x_mid)[0]

    train_data = {}
    val_data = {}

    for key in data.keys():
        train_data[key] = data[key][train_indices]
        val_data[key] = data[key][val_indices]

    return train_data, val_data

def compute_edges_knn(positions, k=10):
    """
    Compute edge indices using k-Nearest Neighbors algorithm with scipy's cKDTree.
    Returns a global edge_index that you can subset later.
    """
    num_nodes = positions.shape[0]
    tree = cKDTree(positions)
    distances, indices = tree.query(positions, k=k + 1)
    indices = indices[:, 1:]  # Exclude self-loops

    row_indices = np.repeat(np.arange(num_nodes), k)
    col_indices = indices.flatten()
    edge_index = np.stack([row_indices, col_indices], axis=0)

    return edge_index.astype(np.int64)

class GraphDataset(Dataset):
    """
    Dataset class for the graph neural network.
    Each item now corresponds to a single node.
    """
    def __init__(self, positions, orientations, properties, k=10):
        self.positions = positions.astype(np.float32)
        self.orientations = orientations.astype(np.float32)
        self.h = properties.astype(np.float32)
        self.k = k
        self.edge_index = compute_edges_knn(self.positions, self.k)

        self.num_nodes = self.positions.shape[0]

    def __len__(self):
        return self.num_nodes

    def __getitem__(self, idx):
        # Return a single node's features
        # Note: We don't return edges here. Edges are global and will be filtered in collate_fn.
        return {
            'x': self.positions[idx],      # position of this node
            'h': self.h[idx],              # property of this node
            'target': self.orientations[idx],
            'node_idx': idx                # keep track of original node index for subgraph creation
        }

def collate_fn(batch):
    """
    Collate function for PyTorch DataLoader.
    We now have a batch of nodes. We need to:
    - Stack their features
    - Determine which edges are internal to this batch
    """
    # Extract fields
    x = torch.stack([torch.from_numpy(item['x']) for item in batch], dim=0)          # [B, pos_dim]
    h = torch.stack([torch.from_numpy(item['h']) for item in batch], dim=0)          # [B, h_dim]
    target = torch.stack([torch.from_numpy(item['target']) for item in batch], dim=0)# [B, target_dim]
    node_indices = np.array([item['node_idx'] for item in batch])

    # We need to determine edges among the selected nodes
    # We'll map the global node indices to batch indices
    node_map = {orig_idx: i for i, orig_idx in enumerate(node_indices)}

    # Access the global edge_index from one of the items
    # They should all have the same dataset structure, so just pick the first
    # Actually, we need a reference to the dataset or precompute edges outside.
    # A neat trick: store global edges in a static variable or pass via closure.
    # Here we assume the dataset has global edges, so let's do this via a hack:
    # We'll store edge_index in the first element's dataset reference if needed.
    # A cleaner approach: don't rely on dataset inside collate_fn;
    # Instead, store a reference outside. But to keep it simple:

    # This requires that 'edge_index' is accessible here. If not, we must
    # pass it into DataLoader or wrap this collate_fn.
    # Let's assume we do something like this:
    # We'll define a closure that holds a reference to the dataset's edge_index.

    # If we can't modify the signature easily:
    #   - Create a custom collate_fn maker that captures a reference to global edge_index.
    # E.g.:
    # def make_collate_fn(edge_index):
    #     def collate_fn(batch):
    #         ...
    #     return collate_fn
    #
    # For clarity, let's assume we do that below and show how to integrate
    # at the bottom in create_dataloaders.

    # Placeholder: We'll assume 'collate_fn' here has a global 'global_edge_index' variable:
    global global_edge_index

    # Filter edges to only those between nodes in this batch
    src = global_edge_index[0]
    dst = global_edge_index[1]

    mask = np.isin(src, node_indices) & np.isin(dst, node_indices)
    src_filtered = src[mask]
    dst_filtered = dst[mask]

    # Map original node IDs to batch node IDs
    src_mapped = [node_map[i] for i in src_filtered]
    dst_mapped = [node_map[i] for i in dst_filtered]
    edge_index = torch.tensor([src_mapped, dst_mapped], dtype=torch.long)

    return h, x, edge_index, target

def create_dataloaders(root_dir, alignment_strength, num_neighbors=10, batch_size=1024, shuffle=True):
    """
    Create data loaders for training and validation sets with batching.
    """
    data = load_dataset(root_dir, alignment_strength)
    train_data, val_data = split_dataset(data)

    train_dataset = GraphDataset(train_data['positions'],
                                 train_data['orientations'],
                                 train_data['properties'],
                                 num_neighbors)

    val_dataset = GraphDataset(val_data['positions'],
                               val_data['orientations'],
                               val_data['properties'],
                               num_neighbors)

    # We need the global edge_index for the collate_fn to work
    # We'll capture it via a closure
    def make_collate_fn(edge_index):
        def wrapped_collate_fn(batch):
            global global_edge_index
            global_edge_index = edge_index  # set global in this scope
            return collate_fn(batch)
        return wrapped_collate_fn

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                              collate_fn=make_collate_fn(train_dataset.edge_index))

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=make_collate_fn(val_dataset.edge_index))

    datasets = {'train': train_dataset, 'val': val_dataset}
    dataloaders = {'train': train_loader, 'val': val_loader}

    return datasets, dataloaders
