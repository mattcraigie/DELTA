import torch
from ..data.dataloading import GraphDataset
import matplotlib.pyplot as plt
from gnnshap.explainer import GNNShapExplainer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from ..utils.shap_analysis import DirectionClassificationWrapper, collate_fn
from ..utils.plotting import save_plot
import yaml

def analyze_shap_vs_distance(explainer, data, max_distance, num_samples, num_explained_galaxies, batch_size=128):
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
                int(idx),
                nsamples=num_samples,
                sampler_name='GNNShapSampler',
                batch_size=batch_size
            )
        except AssertionError as e:
            continue

        # Get linked nodes' positions
        linked_nodes = explanation.sub_edge_index
        source_positions = positions[linked_nodes[0]]  # Assuming linked_nodes[0] provides global indices
        weights = np.abs(explanation.shap_values)

        # filter out outliers (e.g. nodes with very high outlier values)
        outlier_cut = weights < 1e3
        weights = weights[outlier_cut]
        source_positions = source_positions[outlier_cut]

        # Calculate distances
        node_position = positions[idx]  # (2,)
        distances = source_positions - node_position
        distance_norms = np.linalg.norm(distances, axis=1)

        # Bin distances
        bin_index = np.digitize(distance_norms, distance_bins) - 1

        valid_bins = (bin_index >= 0) & (bin_index < num_bins)  # Keep valid indices
        bin_index = bin_index[valid_bins]
        weights = weights[valid_bins]  # Ensure weights align with valid distances

        for b_i, w in zip(bin_index, weights):
            bin_values[b_i] += w
            bin_counts[b_i] += 1

    # Average values in each bin
    mask = bin_counts > 0
    bin_values[mask] /= bin_counts[mask]

    # Define bin centers
    bin_centers = 0.5 * (distance_bins[1:] + distance_bins[:-1])


    return bin_centers, bin_values



def make_plot(bin_centers, mean_bin_values):


    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Mean and std plot
    ax.plot(bin_centers, mean_bin_values, 'b-', label='Mean')

    power_law_m, power_law_c = fit_power_low(bin_centers, mean_bin_values)
    ax.plot(bin_centers, np.exp(power_law_c) * bin_centers ** power_law_m, 'r--', label=f'Power Law Fit: {power_law_m:.2f}')

    ax.set_xlabel('Distance')
    ax.set_ylabel('Mean Importance Value')
    ax.set_title('Mean Importance vs Distance')
    ax.legend()
    plt.semilogy()

    plt.tight_layout()
    return fig


def fit_power_low(x, y):
    """
    Use a log transform to fit a power law to the data with a linear regression.

    Outputs:
    - m: Slope of the fit
    - c: Intercept of the fit

    The slope m is the power law exponent. The intercept c is the log of the prefactor.
    """

    # remove any zeros
    x = x[y > 0]
    y = y[y > 0]

    log_x = np.log(x)
    log_y = np.log(y)

    # Fit a line to the log-transformed data
    A = np.vstack([log_x, np.ones(len(log_x))]).T
    m, c = np.linalg.lstsq(A, log_y, rcond=None)[0]

    return m, c

def run_distance_experiment(model, positions, orientations, properties, k, max_distance, num_samples,
                            num_explained_galaxies, device, analysis_dir, file_name_prefix=None):

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

    bin_centers, mean_bin_values = analyze_shap_vs_distance(explainer, data, max_distance, num_samples,
                                                            num_explained_galaxies)

    fig = make_plot(bin_centers, mean_bin_values)

    save_plot(fig, root_dir=analysis_dir, file_name=file_name_prefix + "_distance_analysis.png")


if __name__ == '__main__':

    import argparse
    import os
    from ..models.vmdn import init_vmdn

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='Run distance analysis')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_distance', type=float, default=50)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--num_galaxies', type=int, default=10000)

    args = parser.parse_args()

    # load yaml config
    with open(os.path.join(args.output_dir, 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    model = init_vmdn(config["model"])
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'model.pth')))
    model.to(device)

    k = config["data"]["num_neighbors"]

    positions = np.load(os.path.join(args.output_dir, 'positions.npy'))
    orientations = np.load(os.path.join(args.output_dir, 'targets.npy'))

    # back to directions
    orientations = np.stack([np.cos(orientations), np.sin(orientations)], axis=1)
    properties = np.load(os.path.join(args.output_dir, 'properties.npy'))

    analysis_dir = os.path.join(args.output_dir, 'distance_analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    run_distance_experiment(model, positions, orientations, properties, k=k, max_distance=args.max_distance, device=device,
                            num_samples=args.num_samples, num_explained_galaxies=args.num_galaxies,
                            analysis_dir=analysis_dir, file_name_prefix='distances')