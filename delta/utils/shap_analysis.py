import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# todo: make this work

def analyze_importance_distance(gnn_model, dataset, num_samples=1000, batch_size=128):
    """
    Analyze the relationship between GNNShap importance values and physical distances
    for a galaxy dataset.

    Parameters:
    -----------
    gnn_model : torch.nn.Module
        The trained GNN model
    dataset : GraphDataset
        Your galaxy dataset containing positions, orientations, and properties
    num_samples : int
        Number of samples for GNNShap
    batch_size : int
        Batch size for processing
    """
    positions = dataset.positions
    num_galaxies = positions.shape[0]
    device = next(gnn_model.parameters()).device

    # Calculate all pairwise distances between galaxies
    distances_matrix = cdist(positions, positions)

    # Define distance bins based on data distribution
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
        nhops=2,  # Adjust based on your model's architecture
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

    # Create main plot
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