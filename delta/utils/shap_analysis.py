import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# todo: make this work
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from gnnshap.explainer import GNNShapExplainer

class DirectionClassificationWrapper(nn.Module):
    def __init__(self, egnn_model, num_classes=8):
        super(DirectionClassificationWrapper, self).__init__()
        self.egnn_model = egnn_model
        self.num_classes = num_classes

    def forward(self, node_features, edge_index, edge_weight=None):
        # Assume node_features = [h, x]
        # Let's say h has dimension self.egnn_model.node_embedding.in_features (num_properties)
        # x has dimension 2 (positions)
        h_dim = self.egnn_model.node_embedding.in_features
        h = node_features[:, :h_dim]
        x = node_features[:, h_dim:h_dim+2]

        # Run the original EGNN model
        h_out, x_out, v_out = self.egnn_model(h, x, edge_index)

        # v_out: (N, 2) unit vectors
        # Compute angles in [0,2π)
        angles = torch.atan2(v_out[:,1], v_out[:,0])
        angles = angles % (2 * np.pi)  # ensure in [0, 2π)

        # Determine class bins
        bin_size = (2 * np.pi) / self.num_classes
        class_ids = torch.floor(angles / bin_size).long()

        # Create logits (N, num_classes)
        # We'll put a large negative number for all classes except the chosen one
        logits = torch.full((v_out.size(0), self.num_classes), -1000.0, device=v_out.device)
        logits[torch.arange(v_out.size(0)), class_ids] = 0.0

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


class GraphDataset(Dataset):
    def __init__(self, positions, orientations, properties, k=10):
        self.positions = positions  # (N, 2)
        self.orientations = orientations  # (N, 2)
        self.h = properties.astype(np.float32)  # (N, F_h)
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

# Suppose you have a pre-trained EGNN model: pretrained_egnn
num_classes = 8
wrapped_model = DirectionClassificationWrapper(pretrained_egnn, num_classes=num_classes)

# Prepare the data
dataset = GraphDataset(positions, orientations, properties, k=10)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=lambda b: collate_fn(b, num_classes=num_classes))
data = next(iter(loader))  # get the single data object

explainer = GNNShapExplainer(
    model=wrapped_model,
    data=data,
    # forward_fn can be default_predict_fn or a custom one if needed,
    # but default should now work since we return logits for multiple classes.
)

explanation = explainer.explain(node_idx=0, nsamples=1000)


def analyze_importance_distance(gnn_model, dataset, num_samples=1000, batch_size=128):
    """
    Analyze the relationship between GNNShap importance values and physical distances
    for a dataset now in PyG Data format.

    Parameters:
    -----------
    gnn_model : torch.nn.Module
        The trained and wrapped GNN model (DirectionClassificationWrapper)
    dataset : torch_geometric.data.Data
        Data object containing x (node_features = [h, x]), edge_index, and y (class labels)
    num_samples : int
        Number of samples for GNNShap
    batch_size : int
        Batch size for processing
    """

    # Extract positions from dataset.x
    # Determine h_dim from the gnn_model
    h_dim = gnn_model.egnn_model.node_embedding.in_features
    positions = dataset.x[:, h_dim:h_dim + 2].cpu().numpy()

    num_galaxies = positions.shape[0]
    device = next(gnn_model.parameters()).device

    # Calculate all pairwise distances between galaxies
    distances_matrix = cdist(positions, positions)

    # Define distance bins
    min_dist = distances_matrix[distances_matrix > 0].min()
    max_dist = distances_matrix.max()
    num_bins = 10
    distance_bins = np.linspace(min_dist, max_dist, num_bins + 1)

    # Initialize arrays for storing results
    bin_values = np.zeros((num_galaxies, num_bins))
    bin_counts = np.zeros((num_galaxies, num_bins))

    # Create GNNShap explainer
    gshap = GNNShapExplainer(
        gnn_model,
        dataset,
        nhops=3,  # Adjust based on your model's architecture
        verbose=0,
        device=device,
        progress_hide=True
    )

    # Analyze each galaxy
    for i in range(num_galaxies):
        # Get explanation for current galaxy
        explanation = gshap.explain(
            i,
            nsamples=num_samples,
            sampler_name='GNNShapSampler',
            batch_size=batch_size
        )

        weights = np.abs(explanation.shap_values)
        # Remove extreme outliers
        weights[weights > np.percentile(weights, 99)] = 0

        # Get distances from current galaxy to all others
        galaxy_distances = distances_matrix[i]

        # Bin the importance values by distance
        for j, (d, w) in enumerate(zip(galaxy_distances, weights)):
            if j == i:  # Skip self-interaction
                continue

            bin_idx = np.digitize(d, distance_bins) - 1
            if 0 <= bin_idx < num_bins:
                bin_values[i, bin_idx] += w
                bin_counts[i, bin_idx] += 1

    # Average values in each bin
    mask = bin_counts > 0
    bin_values[mask] /= bin_counts[mask]

    # Calculate statistics
    mean_bin_values = np.nanmean(bin_values, axis=0)
    std_bin_values = np.nanstd(bin_values, axis=0)

    # Plotting
    bin_centers = 0.5 * (distance_bins[1:] + distance_bins[:-1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 2D histogram
    flattened_bin_values = bin_values.flatten()
    flattened_bin_distances = np.repeat(bin_centers, num_galaxies)
    h = ax1.hist2d(
        flattened_bin_distances,
        flattened_bin_values,
        bins=(10, 10),
        cmap='viridis',
        norm=LogNorm()
    )
    plt.colorbar(h[3], ax=ax1, label='Frequency (log scale)')
    ax1.set_xlabel('Distance')
    ax1.set_ylabel('Importance Value')
    ax1.set_title('Distribution of Importance vs Distance')

    # Mean and std plot
    ax2.plot(bin_centers, mean_bin_values, 'b-', label='Mean')
    ax2.fill_between(
        bin_centers,
        mean_bin_values - std_bin_values,
        mean_bin_values + std_bin_values,
        alpha=0.3,
        color='b'
    )
    ax2.set_xlabel('Distance')
    ax2.set_ylabel('Mean Importance Value')
    ax2.set_title('Mean Importance vs Distance')
    ax2.legend()

    plt.tight_layout()
    return fig, (mean_bin_values, std_bin_values, bin_centers)

#todo: restructure code so that the model runs, saves, and then the experiments run post-training.