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
    positions = data.x[:, 1:].cpu().numpy()

    # Extract the number of galaxies
    num_galaxies = positions.shape[0]

    # Define distance bins
    min_dist = 0
    num_bins = 20
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

    np.save(os.path.join(analysis_dir, 'bin_centers.npy'), bin_centers)
    np.save(os.path.join(analysis_dir, 'mean_bin_values.npy'), mean_bin_values)

    fig = make_plot(bin_centers, mean_bin_values)

    save_plot(fig, root_dir=analysis_dir, file_name=file_name_prefix + "_distance_analysis.png")


def shap_influence_scatter(explainer, data, galaxy_idx, num_samples, batch_size=128):
    """
    Measures SHAP values of galaxies near a given galaxy and maps their influence.

    Parameters:
    - explainer: Explainer object for the GNN model
    - data: Dataset containing node features and edge indices
    - galaxy_idx: Index of the galaxy for which neighbors are analyzed
    - num_samples: Number of samples for SHAP explanation
    - batch_size: Batch size for SHAP explanation

    Returns:
    - scatter_fig: Matplotlib figure object with the scatter plot
    """
    positions = data.x[:, 1:].cpu().numpy()

    print(data.x.shape)
    print(data.x)

    # Get SHAP explanation for the given galaxy
    try:
        explanation = explainer.explain(
            int(galaxy_idx),
            nsamples=num_samples,
            sampler_name='GNNShapSampler',
            batch_size=batch_size
        )
    except AssertionError as e:
        print(f"Failed to compute SHAP explanation for galaxy {galaxy_idx}: {e}")
        return None

    # Extract linked nodes and SHAP values
    linked_nodes = explanation.sub_edge_index
    source_positions = positions[linked_nodes[0]]
    weights = np.abs(explanation.shap_values)

    # Filter out outliers
    valid_mask = weights < 1e3
    weights = weights[valid_mask]
    source_positions = source_positions[valid_mask]

    # save positions and weights in an array together
    source_positions = np.concatenate([source_positions, weights[:, None]], axis=1)

    return source_positions

def run_shapmap_experiment(model, positions, orientations, properties, k, num_samples, device, analysis_dir, file_name_prefix=None):
    """
    Runs an experiment to analyze SHAP values and generate scatter plots for multiple galaxies.

    Parameters:
    - model: The GNN model to analyze
    - positions: Positions of the galaxies
    - orientations: Orientations of the galaxies
    - properties: Properties of the galaxies
    - k: Number of neighbors in the graph
    - num_samples: Number of samples for SHAP explanation
    - num_explained_galaxies: Number of galaxies to analyze
    - device: Torch device for computations
    - analysis_dir: Directory to save results
    - file_name_prefix: Prefix for saved file names
    """
    from gnnshap.explainer import GNNShapExplainer
    from ..data.dataloading import GraphDataset
    from ..utils.shap_analysis import DirectionClassificationWrapper, collate_fn
    import os

    num_classes = 8
    wrapped_model = DirectionClassificationWrapper(model, num_classes=num_classes)

    # Prepare the data
    dataset = GraphDataset(positions, orientations, properties, k)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         collate_fn=lambda b: collate_fn(b, num_classes=num_classes))
    data = next(iter(loader))

    print(data.x.shape)

    # Create GNNShap explainer
    explainer = GNNShapExplainer(
        wrapped_model,
        data,
        nhops=3,  # Adjust based on your model's architecture
        verbose=0,
        device=device,
        progress_hide=True
    )

    # hardcoding in a nice one to visualise because it sits in a subhalo
    point_loc = (944, 976)  # The target point
    distances = np.sum((positions[:, :2] - np.array(point_loc)) ** 2, axis=1)
    galaxy_idx = np.argmin(distances)
    print(positions[galaxy_idx])

    shap_influence = shap_influence_scatter(explainer, data, galaxy_idx=galaxy_idx, num_samples=num_samples)

    # save positions
    np.save(os.path.join(analysis_dir, 'shap_influence.npy'), shap_influence)


if __name__ == '__main__':

    import argparse
    import os
    from ..models.vmdn import init_vmdn

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='Run distance analysis')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_distance', type=float, default=30)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--num_galaxies', type=int, default=10000)
    parser.add_argument('--run_mapping', type=bool, default=False)
    parser.add_argument('--run_distance', type=bool, default=True)

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

    if args.run_mapping:
        run_shapmap_experiment(model, positions, orientations, properties, k=k, num_samples=args.num_samples,
                               device=device, analysis_dir=analysis_dir, file_name_prefix='shapmap')

    if args.run_distance:
        run_distance_experiment(model, positions, orientations, properties, k=k, max_distance=args.max_distance, device=device,
                                num_samples=args.num_samples, num_explained_galaxies=args.num_galaxies,
                                analysis_dir=analysis_dir, file_name_prefix='distances')

