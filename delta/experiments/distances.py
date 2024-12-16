import torch
from ..data.dataloading import GraphDataset
import matplotlib.pyplot as plt
from gnnshap.explainer import GNNShapExplainer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from ..utils.shap_analysis import DirectionClassificationWrapper, collate_fn
from ..utils.plotting import save_plot

def analyze_importance_distance(explainer, positions, max_distance, num_samples=1000, batch_size=128):
    """
    Analyzes the relationship between SHAP importance values and distances between data points.

    Parameters:
    - explainer: Explainer object for the GNN model
    - positions: 2D array-like positions of data points
    - device: Torch device for computations
    - num_samples: Number of samples for SHAP explanation
    - batch_size: Batch size for SHAP explanation
    - max_distance: Maximum distance for the analysis

    Returns:
    - bin_centers: Midpoints of distance bins
    - mean_bin_values: Mean SHAP values in each distance bin
    - std_bin_values: Standard deviation of SHAP values in each distance bin
    """
    # Extract the number of galaxies
    num_galaxies = positions.shape[0]

    # Define distance bins
    min_dist = 0
    num_bins = 10
    distance_bins = np.linspace(min_dist, max_distance, num_bins + 1)

    # Initialize arrays for storing results
    bin_values = np.zeros((num_galaxies, num_bins))
    bin_counts = np.zeros((num_galaxies, num_bins))

    # Setup a KNN model to calculate distances efficiently
    knn = NearestNeighbors(radius=max_distance, metric='euclidean')
    knn.fit(positions)

    for i in range(num_galaxies):
        # Get explanation for the current galaxy
        explanation = explainer.explain(
            i,
            nsamples=num_samples,
            sampler_name='GNNShapSampler',
            batch_size=batch_size
        )

        weights = np.abs(explanation.shap_values)
        # Remove extreme outliers
        weights[weights > np.percentile(weights, 99)] = 0

        # Query distances from the current galaxy to others
        distances, indices = knn.radius_neighbors([positions[i]], radius=max_distance)
        distances, indices = distances[0], indices[0]

        # Map global indices to local indices
        subset = explanation.subset  # Ensure this corresponds to the subset used for `weights`
        global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(subset)}
        filtered_indices = [global_to_local[idx] for idx in indices if idx in global_to_local]

        # Slice weights to match filtered indices
        weights = weights[filtered_indices]

        # Bin the importance values by distance
        for w_idx, d_idx in enumerate(filtered_indices):
            if d_idx == i:  # Skip self-interaction
                continue

            bin_idx = np.digitize(distances[w_idx], distance_bins) - 1
            if 0 <= bin_idx < num_bins:
                bin_values[i, bin_idx] += weights[w_idx]
                bin_counts[i, bin_idx] += 1

    # Average values in each bin
    mask = bin_counts > 0
    bin_values[mask] /= bin_counts[mask]

    # Calculate statistics
    mean_bin_values = np.nanmean(bin_values, axis=0)
    std_bin_values = np.nanstd(bin_values, axis=0)

    # Plotting
    bin_centers = 0.5 * (distance_bins[1:] + distance_bins[:-1])


    return bin_centers, mean_bin_values, std_bin_values


def make_plot(bin_centers, mean_bin_values, std_bin_values):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

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
    return fig


def run_distance_experiment(model, positions, orientations, properties, k, max_distance, device, analysis_dir,
                            file_name_prefix=None):

    num_classes = 8
    wrapped_model = DirectionClassificationWrapper(model, num_classes=num_classes)

    # Prepare the data
    dataset = GraphDataset(positions, orientations, properties, k)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         collate_fn=lambda b: collate_fn(b, num_classes=num_classes))
    data = next(iter(loader))

    # Create GNNShap explainer
    explainer = GNNShapExplainer(
        wrapped_model,
        data,
        nhops=3,  # Adjust based on your model's architecture
        verbose=0,
        device=device,
        progress_hide=True
    )

    bin_centers, mean_bin_values, std_bin_values = analyze_importance_distance(explainer, positions, max_distance)

    fig = make_plot(bin_centers, mean_bin_values, std_bin_values)

    save_plot(fig, root_dir=analysis_dir, file_name=file_name_prefix + "_distance_analysis.png")
