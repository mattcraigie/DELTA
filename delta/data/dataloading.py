import numpy as np
import os as os
from scipy.spatial import cKDTree
import torch
from torch.utils.data import Dataset, DataLoader


def load_dataset(root_dir, alignment_strength):
    """
    Load the preprocessed dataset.
    """

    data_components = ['positions', 'orientations', 'properties']
    data = {}

    for component in data_components:
        path = os.path.join(root_dir, 'preprocessed', '{}_{}.npy'.format(component, alignment_strength))
        if not os.path.exists(path):
            raise FileNotFoundError("The file {} does not exist.".format(path))

        data[component] = np.load(path)

    return data


def split_dataset(data):
    """
    Split the dataset into training and validation sets based on the x-axis. Always split at the midpoint.
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
    """
    num_nodes = positions.shape[0]
    tree = cKDTree(positions)
    distances, indices = tree.query(positions, k=k + 1)
    indices = indices[:, 1:]  # Exclude self-loops (the first neighbor is always the point itself)

    row_indices = np.repeat(np.arange(num_nodes), k)
    col_indices = indices.flatten()
    edge_index = np.stack([row_indices, col_indices], axis=0)

    return edge_index.astype(np.int64)


def collate_fn(batch):
    """
    Collate function for PyTorch DataLoader.
    """
    h = torch.from_numpy(batch[0]['h'])
    x = torch.from_numpy(batch[0]['x'])
    target = torch.from_numpy(batch[0]['target'])
    edge_index = torch.from_numpy(batch[0]['edge_index'])

    return h, x, edge_index, target


class GraphDataset(Dataset):
    """
    Dataset class for the graph neural network.
    """
    def __init__(self, positions, orientations, properties, k=10):
        self.positions = positions
        self.orientations = orientations
        self.h = properties.astype(np.float32)
        self.k = k
        self.edge_index = compute_edges_knn(self.positions, self.k)

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        return {
            'x': self.positions,
            'edge_index': self.edge_index,
            'h': self.h,
            'target': self.orientations
        }


def create_dataloaders(root_dir, alignment_strength, num_neighbors=10):
    """
    Create data loaders for training and validation sets.
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

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    datasets = {'train': train_dataset, 'val': val_dataset}
    dataloaders = {'train': train_loader, 'val': val_loader}

    return datasets, dataloaders