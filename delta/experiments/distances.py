import torch
from ..data.dataloading import GraphDataset
import matplotlib.pyplot as plt
from gnnshap.explainer import GNNShapExplainer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from ..utils.shap_analysis import DirectionClassificationWrapper, collate_fn
from ..utils.plotting import save_plot
import yaml

def analyze_shap_vs_distance(explainer, data, max_distance=50, num_samples=1000, batch_size=128,
                             num_explained_galaxies=1000):
    """
    Analyzes the relationship between SHAP importance values and distances between data points.

    Parameters:
    - explainer: Explainer object for the GNN model
    - data: Dataset containing node features and edge indices
    - device: Torch device for computations
    - num_samples: Number of samples for SHAP explanation
    - batch_size: Batch size for SHAP explanation
    - max_distance: Maximum distance for the analysis
    - num_explained_galaxies: Number of galaxies to explain

    Returns:
    - bin_centers: Midpoints of distance bins
    - mean_bin_values: Mean SHAP values in each distance bin
    - std_bin_values: Standard deviation of SHAP values in each distance bin
    """
    positions = data.x[:, -2:].cpu().numpy()

    # Extract the number of galaxies
    num_galaxies = positions.shape[0]

    # Define distance bins
    min_dist = 0
    num_bins = 50
    distance_bins = np.linspace(min_dist, max_distance, num_bins + 1)

    # Initialize arrays for storing results
    bin_values = np.zeros((num_bins,))
    bin_counts = np.zeros((num_bins,))

    test_gal_indices = np.arange(num_galaxies)
    np.random.shuffle(test_gal_indices)

    for i, idx in enumerate(test_gal_indices[:num_explained_galaxies]):
        if i % 100 == 0:
            print(f"Processing galaxy {i}/{num_explained_galaxies}")

        try:
            explanation = explainer.explain(
                idx,
                nsamples=num_samples,
                sampler_name='GNNShapSampler',
                batch_size=batch_size
            )
        except AssertionError as e:
            continue

        # Get linked nodes' positions
        linked_nodes = explainer.sub_edge_index
        source_positions = positions[linked_nodes[0]]  # Assuming linked_nodes[0] provides global indices
        weights = explainer.shap_value

        # Calculate distances
        node_position = positions[idx]  # (2,)
        distances = source_positions - node_position
        distance_norms = np.linalg.norm(distances, axis=1)

        # Bin distances
        bin_index = np.digitize(distance_norms, distance_bins) - 1
        for b_i, w in zip(bin_index, weights):
            bin_values[b_i] += np.abs(w)
            bin_counts[b_i] += 1

    # Average values in each bin
    mask = bin_counts > 0
    bin_values[mask] /= bin_counts[mask]

    # Calculate statistics
    mean_bin_values = np.nanmean(bin_values[mask])
    std_bin_values = np.nanstd(bin_values[mask])

    # Define bin centers
    bin_centers = 0.5 * (distance_bins[1:] + distance_bins[:-1])

    return bin_centers, mean_bin_values, std_bin_values



def make_plot(bin_centers, mean_bin_values, std_bin_values):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Mean and std plot
    ax.plot(bin_centers, mean_bin_values, 'b-', label='Mean')
    # ax.fill_between(
    #     bin_centers,
    #     mean_bin_values - std_bin_values,
    #     mean_bin_values + std_bin_values,
    #     alpha=0.3,
    #     color='b'
    # )
    ax.set_xlabel('Distance')
    ax.set_ylabel('Mean Importance Value')
    ax.set_title('Mean Importance vs Distance')
    ax.legend()
    plt.semilogy()

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

    bin_centers, mean_bin_values, std_bin_values = analyze_importance_distance(explainer, data, max_distance)

    fig = make_plot(bin_centers, mean_bin_values, std_bin_values)

    save_plot(fig, root_dir=analysis_dir, file_name=file_name_prefix + "_distance_analysis.png")


if __name__ == '__main__':

    import argparse
    import os
    from ..models.vmdn import init_vmdn

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='Run distance analysis')
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()

    # load yaml config
    with open(os.path.join(args.output_dir, 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    model = init_vmdn(config["model"])
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'model.pth')))
    model.to(device)

    positions = np.load(os.path.join(args.output_dir, 'positions.npy'))
    orientations = np.load(os.path.join(args.output_dir, 'targets.npy'))

    # back to directions
    orientations = np.stack([np.cos(orientations), np.sin(orientations)], axis=1)
    properties = np.load(os.path.join(args.output_dir, 'properties.npy'))

    analysis_dir = os.path.join(args.output_dir, 'distance_analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    run_distance_experiment(model, positions, orientations, properties, k=10, max_distance=50, device=device,
                            analysis_dir=analysis_dir, file_name_prefix='distances')