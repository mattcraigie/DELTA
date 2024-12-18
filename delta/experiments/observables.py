import torch
import numpy as np
import os
from ..data.dataloading import create_dataloaders, load_dataset, split_dataset
from ..models.vmdn import init_vmdn
from ..training.train import train_model
from ..utils.utils import get_model_predictions, signed_to_unsigned_angle, angle_from_trig
from ..utils.plotting import plot_results
from torch.utils.data import DataLoader
from ..data.dataloading import collate_fn

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
    positions_train = datasets['train'].positions
    positions_val = datasets['val'].positions

    # Extract the original properties from the train set (we assume train and val have same shape properties)
    original_properties_train = datasets['train'].h
    original_properties_val = datasets['val'].h

    # Create informative and uninformative observables for train
    informative_obs_train = make_informative_observable(positions_train)
    uninformative_obs_train = make_uninformative_observable(original_properties_train)
    datasets['train'].h = np.column_stack((informative_obs_train, uninformative_obs_train))

    # Create informative and uninformative observables for val
    informative_obs_val = make_informative_observable(positions_val)
    uninformative_obs_val = make_uninformative_observable(original_properties_val)
    datasets['val'].h = np.column_stack((informative_obs_val, uninformative_obs_val))


def run_observables_experiment(config):
    """
    Run an observables test of the model using the updated workflows.
    """

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    analysis_name = config["analysis"]["name"]
    output_dir = config["analysis"]["output_dir"]
    os.makedirs(os.path.join(output_dir, analysis_name), exist_ok=True)

    data_dir = config["data"]["data_root"]
    alignment_strength = config["data"]["alignment_strength"]
    num_neighbors = config["data"]["num_neighbors"]

    # Load datasets and dataloaders
    datasets, dataloaders = create_dataloaders(data_dir, alignment_strength, num_neighbors)

    # Add observables to datasets
    add_observables_to_datasets(datasets)
    train_loader = DataLoader(datasets['train'], batch_size=1, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(datasets['val'], batch_size=1, shuffle=False, collate_fn=collate_fn)
    dataloaders = {'train': train_loader, 'val': val_loader}

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

    # Save results
    analysis_dir = os.path.join(output_dir, analysis_name)
    torch.save(model.state_dict(), os.path.join(analysis_dir, "model.pth"))


    # Plot the results
    plot_results(losses, predictions, targets, analysis_dir, file_name_prefix='data')

    # Repeat with the fully aligned data
    alignment_strength = 1.0
    dataset_full, _ = create_dataloaders(data_dir, alignment_strength, num_neighbors)
    targets_full = dataset_full['val'].orientations
    targets_full = angle_from_trig(targets_full[:, 0], targets_full[:, 1])[:, None]
    targets_full = signed_to_unsigned_angle(targets_full)

    torch.save(targets_full, os.path.join(analysis_dir, "targets_true.pth"))

    plot_results(losses, predictions, targets_full, analysis_dir, file_name_prefix="true")

    # Permutation Experiment
    print("Running permutation experiment...")
    val_h_original = datasets['val'].h.copy()
    num_columns = val_h_original.shape[1]

    for i in range(num_columns):
        # Permute the observable
        observable = datasets['val'].h[:, i]
        datasets['val'].h[:, i] = np.random.permutation(observable)

        # Rebuild val_loader to reflect changed data if needed
        predictions_permuted, _ = get_model_predictions(model, dataloaders['val'], device)

        # Plot results
        plot_results(None, predictions_permuted, targets, analysis_dir,
                     file_name_prefix=f"{analysis_name}_permuted_{i}")
        plot_results(None, predictions_permuted, targets_full, analysis_dir,
                     file_name_prefix=f"{analysis_name}_permuted_{i}_full")

        datasets['val'].h = val_h_original.copy()

    print("Permutation experiment complete.")

    # Ablation Experiment

    # print("Running ablation experiment...")
    # original_train_h = datasets['train'].h.copy()
    # original_val_h = datasets['val'].h.copy()
    #
    # for i in range(num_columns):
    #     # Remove the observable column i
    #     datasets['train'].h = np.delete(datasets['train'].h, i, axis=1)
    #     datasets['val'].h = np.delete(datasets['val'].h, i, axis=1)
    #
    #     # Retrain the model with one less property
    #     if "num_properties" in config["model"]:
    #         config["model"]["num_properties"] = datasets['train'].h.shape[1]
    #
    #     model_ablated = init_vmdn(config["model"])
    #     model_ablated.to(device)
    #
    #     # If pretrain is still desired for each ablation (you may decide otherwise):
    #     if config["training"].get("pretrain", False):
    #         train_model(model_ablated.compression_network.egnn, dataloaders['train'], dataloaders['val'],
    #                     pretrain_epochs, pretrain_lr, device)
    #
    #     model_ablated, losses_ablated = train_model(model_ablated, dataloaders['train'], dataloaders['val'],
    #                                                 train_epochs, train_lr, device)
    #
    #     # Get predictions
    #     predictions_ablated, _ = get_model_predictions(model_ablated, dataloaders['val'], device)
    #
    #     # Plot the results
    #     plot_results(losses_ablated, predictions_ablated, targets, analysis_dir,
    #                  file_name_prefix=f"{analysis_name}_ablated_{i}")
    #     plot_results(losses_ablated, predictions_ablated, targets_full, analysis_dir,
    #                  file_name_prefix=f"{analysis_name}_ablated_{i}_full")
    #
    #     # Reset the dataset to the original state for the next iteration
    #     datasets['train'].h = original_train_h.copy()
    #     datasets['val'].h = original_val_h.copy()
    #
    # print("Ablation experiment complete.")
    # print("Observables experiment finished.")

