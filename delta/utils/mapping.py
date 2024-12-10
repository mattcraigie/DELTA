import torch
from .utils import get_model_predictions
import matplotlib.pyplot as plt
import argparse
from .plotting import save_plot
import numpy as np
import os
import yaml
from ..data.dataloading import create_dataloaders

# todo: I need some way of getting the kappa output from the model, so I can show that the poorly predicted points are
# the ones with low kappa. Probably just rewrite the get_model_predictions function to return the kappa values as well.


def create_prediction_map(positions, predictions, targets, targets_full, root_dir, file_name_prefix=None, map_type=None,
                          mask_edges=None):

    if mask_edges is None:
        mask_edges = (520, 540, 20, 40)

    mask = (
        (positions[:, 0] > mask_edges[0])
        & (positions[:, 0] < mask_edges[1])
        & (positions[:, 1] > mask_edges[2])
        & (positions[:, 1] < mask_edges[3])
    )

    positions = positions[mask]

    if map_type is None:
        map_type = "x_component"

    if map_type == 'x_component':
        # use the inbuilt map
        targets, predictions, targets_full = map(lambda x: np.cos(x), [targets, predictions, targets_full])
        cmap = 'bwr'
    elif map_type == 'y_component':
        targets, predictions, targets_full = map(lambda x: np.sin(x), [targets, predictions, targets_full])
        cmap = 'bwr'
    elif map_type == 'angle':
        cmap = 'twilight'
    else:
        raise ValueError(f"Map type {map_type} not recognized.")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # training targets
    axes[0].scatter(positions[:, 0], positions[:, 1], s=10, c=targets[mask], cmap=cmap)
    axes[0].set_title("Training Targets")

    # model predictions
    axes[1].scatter(positions[:, 0], positions[:, 1], s=10, c=predictions[mask], cmap=cmap)
    axes[1].set_title("Model Predictions")

    # true targets
    axes[2].scatter(positions[:, 0], positions[:, 1], s=10, c=targets_full[mask], cmap=cmap)
    axes[2].set_title("True Targets")

    # prediction error from true targets
    abs_err = np.abs(predictions[mask] - targets_full[mask])
    axes[3].scatter(positions[:, 0], positions[:, 1], s=10, c=abs_err, cmap='cool')
    axes[3].set_title("Prediction Error")

    # Save plot
    if file_name_prefix is None:
        file_name = "prediction_map.png"
    else:
        file_name = f"{file_name_prefix}_prediction_map.png"

    return save_plot(fig, root_dir=root_dir, file_name=file_name)


if __name__ == '__main__':

    # argparse for model path
    parser = argparse.ArgumentParser(description='Create prediction map')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--mask_edges', type=str, default='520,540,20,40')

    args = parser.parse_args()

    # config
    config = torch.load(os.path.join(args.output_dir, 'config.pth'))

    # load model
    predictions = torch.load(os.path.join(args.output_dir, 'predictions.pth'))
    targets = torch.load(os.path.join(args.output_dir, 'targets.pth'))
    targets_full = torch.load(os.path.join(args.output_dir, 'targets_true.pth'))

    # data access
    data_dir = config["data"]["data_root"]
    alignment_strength = config["data"]["alignment_strength"]
    num_neighbors = config["data"]["num_neighbors"]
    datasets, _ = create_dataloaders(data_dir, alignment_strength, num_neighbors)
    positions = datasets['val'].positions

    # create prediction map
    plots_path = os.path.join(args.output_dir, 'plots')

    mask_edges = list(map(int, args.mask_edges.split(',')))

    create_prediction_map(positions, predictions, targets, targets_full, plots_path, 'prediction_map',
                          mask_edges=mask_edges)

