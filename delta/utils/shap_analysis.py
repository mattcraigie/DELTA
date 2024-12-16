import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from .utils import signed_to_unsigned_angle
from ..data.dataloading import compute_edges_knn


class DirectionClassificationWrapper(nn.Module):
    def __init__(self, model, num_classes=8, num_properties=1):
        super(DirectionClassificationWrapper, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.num_properties = num_properties

    def forward(self, node_features, edge_index, edge_weight=None):
        # Assume node_features = [h, x]
         # x has dimension 2 (positions)
        node_features = node_features.to(self.model.device)
        edge_index = edge_index.to(self.model.device)


        h = node_features[:, :self.num_properties]
        x = node_features[:, self.num_properties:self.num_properties+2]

        # Run the original EGNN model
        mu, kappa = self.model(h, x, edge_index)

        # v_out: (N, 2) unit vectors
        # Compute angles in [0,2π)
        angles = signed_to_unsigned_angle(mu)

        # Determine class bins
        bin_size = (2 * np.pi) / self.num_classes
        class_ids = torch.floor(angles / bin_size).long()
        class_ids = class_ids.squeeze(-1)

        # Create logits (N, num_classes)
        # We'll put a large negative number for all classes except the chosen one
        logits = torch.full((mu.size(0), self.num_classes), -1000.0, device=self.model.device)
        logits[torch.arange(mu.size(0)), class_ids] = 0.0

        return logits


def compute_direction_classes(target, num_classes=8):
    # target shape: (N, 2), a unit vector direction
    vx, vy = target[:, 0], target[:, 1]
    angles = np.arctan2(vy, vx)
    angles = np.mod(angles, 2 * np.pi)
    bin_size = (2 * np.pi) / num_classes
    classes = np.floor(angles / bin_size).astype(np.int64)
    return classes


def collate_fn(batch, num_classes=8):
    """
    Collate function to produce data.x as concatenated [h, x],
    and data.y as direction classes.
    """
    # batch is a list of items; we have __len__=1, so batch[0]
    data_item = batch[0]
    h = torch.from_numpy(data_item['h'])  # shape (N, F_h)
    x = torch.from_numpy(data_item['x'])  # shape (N, 2)
    target = torch.from_numpy(data_item['target'])  # shape (N, 2)
    edge_index = torch.from_numpy(data_item['edge_index'])  # shape (2, E)

    # Combine h and x into a single node_features tensor
    node_features = torch.cat([h, x], dim=-1)  # shape (N, F_h+2)

    # Convert target directions into classes
    target_classes = torch.from_numpy(compute_direction_classes(target.numpy(), num_classes=num_classes))  # shape (N,)

    # Create a PyG-like data object
    # If using GNNShapExplainer, it expects .x, .edge_index, .y fields in a torch_geometric.data.Data
    data = Data(x=node_features, edge_index=edge_index, y=target_classes)
    return data

#todo: restructure code so that the model runs, saves, and then the experiments run post-training.