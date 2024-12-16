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
    num_bins = 50
    distance_bins = np.linspace(min_dist, max_distance, num_bins + 1)

    # Initialize arrays for storing results
    bin_values = np.zeros((num_galaxies, num_bins))
    bin_counts = np.zeros((num_galaxies, num_bins))

    # Setup a KNN model to calculate distances efficiently
    knn = NearestNeighbors(radius=max_distance, metric='euclidean')
    knn.fit(positions)

    for i in range(10000):
        if i % 100 == 0:
            print(i)

        try:
            explanation = explainer.explain(
                i,
                nsamples=num_samples,
                sampler_name='GNNShapSampler',
                batch_size=batch_size
            )
        except AssertionError:
            continue


        global_to_local = {g_id: i for i, g_id in enumerate(explanation.sub_nodes)}

        num_edges = explanation.sub_edge_index.shape[1]
        local_edge_index = np.zeros_like(explanation.sub_edge_index)

        for e in range(num_edges):
            src_g = explanation.sub_edge_index[0, e]
            dst_g = explanation.sub_edge_index[1, e]

            # Map global IDs to local indices
            local_edge_index[0, e] = global_to_local[src_g]
            local_edge_index[1, e] = global_to_local[dst_g]

        # Aggregate edge-level shap_values to node-level in the subgraph
        src_nodes = local_edge_index[0]
        dst_nodes = local_edge_index[1]

        sub_node_shap_values = np.zeros(len(explanation.sub_nodes))
        np.add.at(sub_node_shap_values, src_nodes, np.abs(explanation.shap_values))
        np.add.at(sub_node_shap_values, dst_nodes, np.abs(explanation.shap_values))

        # Map to global indexing
        global_node_ids = explanation.sub_nodes
        global_shap_values = np.zeros(num_galaxies)
        global_shap_values[global_node_ids] = sub_node_shap_values

        # KNN query on global positions
        distances, indices = knn.radius_neighbors([positions[i]], radius=max_distance)
        distances, indices = distances[0], indices[0]

        # Now index global_shap_values by indices
        weights = global_shap_values[indices]

        # Bin the importance values by distance
        for w_idx, d_idx in enumerate(indices):
            if d_idx == i:
                continue
            bin_idx = np.digitize(distances[w_idx], distance_bins) - 1
            if 0 <= bin_idx < num_bins:

                # avoid outliers
                abs_weight = np.abs(weights[w_idx])
                if abs_weight > 1:
                    continue

                bin_values[i, bin_idx] += abs_weight
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

    bin_centers, mean_bin_values, std_bin_values = analyze_importance_distance(explainer, positions, max_distance)

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

    config = torch.load(os.path.join(args.output_dir, 'config.pth'))

    model = init_vmdn(config["model"])
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'model.pth')))
    model.to(device)

    positions = torch.load(os.path.join(args.output_dir, 'positions.pth'))
    orientations = torch.load(os.path.join(args.output_dir, 'targets.pth'))
    # back to directions
    orientations = np.stack([np.cos(orientations), np.sin(orientations)], axis=1)
    properties = torch.load(os.path.join(args.output_dir, 'targets_true.pth'))

    analysis_dir = os.path.join(args.output_dir, 'distance_analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    run_distance_experiment(model, positions, orientations, properties, k=10, max_distance=50, device=device,
                            analysis_dir=analysis_dir)