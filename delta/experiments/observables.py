import torch
import numpy as np
import os
from ..data.dataloading import create_dataloaders, load_dataset, split_dataset
from ..models.vmdn import init_vmdn
from ..training.train import train_model
from ..utils.utils import get_model_predictions, signed_to_unsigned_angle, angle_from_trig, get_improvement_percentage
from ..utils.plotting import plot_results, save_plot
from torch.utils.data import DataLoader
from ..data.dataloading import collate_fn
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import time

import torch
import torch.nn.functional as F

import torch
import numpy as np
from scipy.ndimage import gaussian_filter

def assign_density_to_galaxies(galaxy_positions, sigma, grid_size):
    """
    Assign a smoothed density value to each galaxy based on its position.

    Parameters:
    - galaxy_positions (torch.Tensor): Tensor of shape (num_galaxies, 2) containing x and y positions.
    - sigma (float): Standard deviation for Gaussian smoothing.
    - grid_size (int): Number of grid cells along each axis.

    Returns:
    - densities (torch.Tensor): Tensor of shape (num_galaxies,) containing the density value for each galaxy.
    """

    # Determine the min and max positions
    min_pos = galaxy_positions.min(axis=0)
    max_pos = galaxy_positions.max(axis=0)

    # Create a 2D histogram (density grid) of the galaxy positions
    H, xedges, yedges = np.histogram2d(
        galaxy_positions[:, 0], galaxy_positions[:, 1],
        bins=grid_size,
        range=[[min_pos[0], max_pos[0]], [min_pos[1], max_pos[1]]]
    )

    # Apply Gaussian smoothing to the density grid
    H_smoothed = gaussian_filter(H, sigma=sigma)

    # Normalize the smoothed density grid to mean 0 and standard deviation 1
    H_normalized = (H_smoothed - H_smoothed.mean()) / H_smoothed.std()

    # Assign density values to each galaxy based on its position
    # Find the bin indices for each galaxy
    x_indices = np.digitize(galaxy_positions[:, 0], xedges) - 1
    y_indices = np.digitize(galaxy_positions[:, 1], yedges) - 1

    # Ensure indices are within valid range
    x_indices = np.clip(x_indices, 0, grid_size - 1)
    y_indices = np.clip(y_indices, 0, grid_size - 1)

    # Retrieve the density value for each galaxy
    densities = H_normalized[x_indices, y_indices]

    # Convert densities back to a PyTorch tensor
    densities_tensor = torch.from_numpy(densities).float()

    return densities_tensor


def make_informative_observable(positions, ):
    """
    Create an informative observable based on the properties.
    It is standard normal noise + scaled properties.
    """
    densities = assign_density_to_galaxies(positions, sigma=1, grid_size=200)
    return densities


def make_uninformative_observable(properties):
    """
    Create an uninformative observable. It is just standard normal noise.
    """
    uninformative_observable = np.random.normal(0, 1, len(properties))
    return uninformative_observable.astype(np.float32)


def add_observables_to_datasets(datasets):
    """
    Add observables to the datasets in-place. This assumes that datasets['train']
    and datasets['val'] have a 'h' attribute representing the properties.
    """
    # Extract the original properties from the train set (we assume train and val have same shape properties)
    original_properties_train = datasets['train'].h
    original_properties_val = datasets['val'].h

    # Create informative and uninformative observables for train
    uninformative_obs_train = make_uninformative_observable(original_properties_train)
    datasets['train'].h = np.column_stack((original_properties_train, uninformative_obs_train))

    # Create informative and uninformative observables for val
    uninformative_obs_val = make_uninformative_observable(original_properties_val)
    datasets['val'].h = np.column_stack((original_properties_val, uninformative_obs_val))


def plot_swarm(scores_dict,
               analysis_dir,
               y_label='% Improvement',
               title='Swarm Plot',
               fname=None,
               threshold=None):
    """
    Plots a swarm plot based on the difference from the 'base' scores.
    Points are colored based on their corresponding base scores.
    If threshold is set, only baseline scores >= threshold are used.

    Parameters:
    - scores_dict: dict
        A dictionary where keys are categories (e.g., 'base' and other permutations)
        and values are lists of scores.
        Must contain the key 'base' for baseline scores.
    - analysis_dir: str
        Directory where the plot will be saved.
    - y_label: str
        Label for the y-axis.
    - title: str
        Title of the plot.
    - fname: str or None
        Filename to save the plot. If None, defaults to 'observables.png'.
    - threshold: float or None
        If provided, baseline scores below this value are excluded from the
        differences/plot.
    """

    if fname is None:
        fname = 'observables.png'

    # Ensure we have a 'base' in the dictionary
    if 'base' not in scores_dict:
        raise ValueError("scores_dict must contain a 'base' key with baseline scores.")

    base_scores = np.array(scores_dict['base'])

    # Identify valid indices based on threshold
    if threshold is not None:
        valid_indices = [i for i, val in enumerate(base_scores) if val >= threshold]
    else:
        valid_indices = range(len(base_scores))

    # Build a new dictionary of differences from base
    differences_dict = {}
    for category, values in scores_dict.items():
        if category == 'base':
            continue  # Skip the base
        category_arr = np.array(values)
        diff_vals = category_arr[valid_indices] - base_scores[valid_indices]
        differences_dict[category] = diff_vals

    # Map categories to x positions
    x_positions = {cat: i for i, cat in enumerate(differences_dict.keys())}

    # Prepare for scatter plotting
    jittered_x = []
    y_values = []
    colors = []
    cmap = get_cmap('viridis')  # Use a gradient colormap

    for cat, x in x_positions.items():
        vals = differences_dict[cat]
        jittered_x.extend(x + np.random.uniform(-0.1, 0.1, size=len(vals)))
        y_values.extend(vals)

        # Assign colors based on the corresponding base scores
        base_vals = base_scores[valid_indices]
        colors.extend(cmap((base_vals - np.min(base_vals)) / (np.max(base_vals) - np.min(base_vals))))

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(jittered_x, y_values, alpha=0.7, edgecolor='k', linewidth=0.5, c=colors)
    plt.xticks(list(x_positions.values()), list(x_positions.keys()))
    plt.xlim(-0.5, len(differences_dict) - 0.5)
    plt.xlabel('Category')
    plt.ylabel(y_label)
    plt.title(title)

    # Add colorbar for base scores
    cbar = plt.colorbar(scatter, label='Base Score')
    cbar.set_label('Base Score')

    # Add annotations for mean and std deviation (based on differences)
    annotation_text = ""
    for cat, vals in differences_dict.items():
        mean_val = np.mean(vals) if len(vals) > 0 else float('nan')
        std_val = np.std(vals) if len(vals) > 0 else float('nan')
        annotation_text += f"{cat}: {mean_val:.2f} ± {std_val:.2f}\n"

    plt.gca().text(
        0.95, 0.05, annotation_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
    )

    # Save the plot (assuming save_plot is defined elsewhere)
    save_plot(plt.gcf(), root_dir=analysis_dir, file_name=fname)



def run_observables_experiment(config):
    """
    Run an observables test of the model using the updated workflows.
    """

    # hardcoded in for now
    repeats = 10
    num_permutations = 1  # Number of sub-repeats per permutation

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    analysis_name = config["analysis"]["name"]
    output_dir = config["analysis"]["output_dir"]
    os.makedirs(os.path.join(output_dir, analysis_name), exist_ok=True)

    data_dir = config["data"]["data_root"]
    alignment_strength = config["data"]["alignment_strength"]
    num_neighbors = config["data"]["num_neighbors"]

    analysis_dir = os.path.join(output_dir, analysis_name)

    # Load datasets and dataloaders
    datasets, dataloaders = create_dataloaders(data_dir, alignment_strength, num_neighbors)

    # Add observables to datasets
    add_observables_to_datasets(datasets)
    train_loader = DataLoader(datasets['train'], batch_size=1, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(datasets['val'], batch_size=1, shuffle=False, collate_fn=collate_fn)
    dataloaders = {'train': train_loader, 'val': val_loader}

    num_columns = datasets['train'].h.shape[1]

    scores_dict_data = {'base': []} | {i: [] for i in range(num_columns)}
    scores_dict_full = {'base': []} | {i: [] for i in range(num_columns)}

    start_time = time.time()

    for repeat in range(repeats):
        print(f"Repeat {repeat + 1}/{repeats}")

        # Initialize the model
        if "num_properties" in config["model"]:
            config["model"]["num_properties"] = dataloaders['train'].dataset.h.shape[1]
        model = init_vmdn(config["model"])
        model.to(device)

        print("Training model...")

        # Possibly pre-train the model if specified
        if config["training"].get("pretrain", False):
            pretrain_epochs = config["training"]["pretrain_epochs"]
            pretrain_lr = config["training"]["pretrain_learning_rate"]
            train_model(model.compression_network.egnn, dataloaders['train'], dataloaders['val'],
                        pretrain_epochs, pretrain_lr, device)

        # Train the model
        train_epochs = config["training"]["train_epochs"]
        train_lr = config["training"]["train_learning_rate"]
        model, losses = train_model(model, dataloaders['train'], dataloaders['val'], train_epochs, train_lr, device)

        # Get predictions
        predictions, targets = get_model_predictions(model, dataloaders['val'], device)
        base_error_data = get_improvement_percentage(predictions, targets)
        scores_dict_data['base'].append(base_error_data)


        # Repeat with the fully aligned data
        alignment_strength = 'full'
        dataset_full, _ = create_dataloaders(data_dir, alignment_strength, num_neighbors)
        targets_full = dataset_full['val'].orientations
        targets_full = angle_from_trig(targets_full[:, 0], targets_full[:, 1])[:, None]
        targets_full = signed_to_unsigned_angle(targets_full)
        base_error_full = get_improvement_percentage(predictions, targets_full)
        scores_dict_full['base'].append(base_error_full)

        # Permutation Experiment
        val_h_original = datasets['val'].h.copy()

        for i in range(num_columns):
            perm_error_data_list = []
            perm_error_full_list = []

            for _ in range(num_permutations):
                # Permute the observable
                observable = datasets['val'].h[:, i]
                datasets['val'].h[:, i] = np.random.permutation(observable)

                # Rebuild val_loader to reflect changed data if needed
                predictions_permuted, _ = get_model_predictions(model, dataloaders['val'], device)

                # Get error scores for the current permutation
                perm_error_i_data = get_improvement_percentage(predictions_permuted, targets)
                perm_error_data_list.append(perm_error_i_data)

                perm_error_i_full = get_improvement_percentage(predictions_permuted, targets_full)
                perm_error_full_list.append(perm_error_i_full)

                # Reset the dataset to its original state
                datasets['val'].h = val_h_original.copy()

            # Compute the median error scores over all sub-repeats
            median_perm_error_data = np.median(perm_error_data_list)
            median_perm_error_full = np.median(perm_error_full_list)

            # Store the median scores in the respective dictionaries
            scores_dict_data[i].append(median_perm_error_data)
            scores_dict_full[i].append(median_perm_error_full)

    # save the scores_dict_data and scores_dict_full
    np.save(os.path.join(analysis_dir, "scores_dict_data.npy"), scores_dict_data)
    np.save(os.path.join(analysis_dir, "scores_dict_full.npy"), scores_dict_full)

    plot_swarm(scores_dict_data, analysis_dir, y_label='% Error', title='Permutation Experiment - Data', fname='observables_data.png')
    plot_swarm(scores_dict_full, analysis_dir, y_label='% Error', title='Permutation Experiment - Full', fname='observables_full.png')

    print("Permutation experiment complete.")

    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.2f} seconds.")

    # save time to txt file
    with open(os.path.join(analysis_dir, "time.txt"), 'w') as file:
        file.write(f"Time taken: {end_time - start_time:.2f} seconds.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run observables experiment')

    # output_dir argument
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=None)


    args = parser.parse_args()

    # load dicts and plot
    scores_dict_data = np.load(os.path.join(args.output_dir, "scores_dict_data.npy"), allow_pickle=True).item()
    scores_dict_full = np.load(os.path.join(args.output_dir, "scores_dict_full.npy"), allow_pickle=True).item()

    plot_swarm(scores_dict_data, args.output_dir, y_label='% Error', title='Permutation Experiment - Data',
               fname='observables_data.png', threshold=args.threshold)
    plot_swarm(scores_dict_full, args.output_dir, y_label='% Error', title='Permutation Experiment - Full',
               fname='observables_full.png', threshold=args.threshold)
