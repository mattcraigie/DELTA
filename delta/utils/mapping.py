import torch
from .utils import angular_differences, angular_mean_with_error
import matplotlib.pyplot as plt
import argparse
from .plotting import save_plot
import numpy as np
import os
import yaml
from ..data.dataloading import create_dataloaders

# todo: update these functions with the new ones from my jupyter notebook

def make_prediction_maps(positions, predictions, targets, targets_full, cmap):
    """
    Create a figure with the prediction maps for training targets, model predictions, and true targets.
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # training targets
    axes[0].scatter(positions[:, 0], positions[:, 1], s=10, c=targets, cmap=cmap)
    axes[0].set_title("Training Targets")

    # model predictions
    axes[1].scatter(positions[:, 0], positions[:, 1], s=10, c=predictions, cmap=cmap)
    axes[1].set_title("Model Predictions")

    # true targets
    axes[2].scatter(positions[:, 0], positions[:, 1], s=10, c=targets_full, cmap=cmap)
    axes[2].set_title("True Targets")

    return fig


def error_heatmap(ax, prediction_error, kappa, x_variable='Prediction Error', y_variable='Model Kappa', bins=50,
                             hist_range=None, pmax=None, cmap='Blues'):
    """
    Create a heatmap of the conditional distribution P(y|x) of model kappa given prediction error.
    """

    # Compute joint histogram
    joint_histogram, xedges, yedges = np.histogram2d(prediction_error.squeeze(1), kappa.squeeze(1), bins=bins, range=hist_range, density=True)

    # Normalize joint histogram to form joint PDF
    joint_pdf = joint_histogram / joint_histogram.sum()

    # Conditional distribution P(y|x)
    sum_x = joint_pdf.sum(axis=1)[:, np.newaxis]
    py_given_x = joint_pdf / (sum_x + (sum_x == 0))

    # Determine maximum value for consistent color scale
    if pmax is None:
        pmax = py_given_x.max()

    # Setup figure and plot

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    ax.imshow(py_given_x.T, origin='lower', extent=extent, aspect='auto', cmap=cmap, vmax=pmax)
    ax.set_xlabel(x_variable)
    ax.set_ylabel(y_variable)


def error_means(ax, prediction, target, n_bins=20, n_bootstrap=100, root_dir=None, file_name=None):
    """
    Plot angular means of target binned by prediction with bootstrap error bars, wrapping error bars within the range
    [-π, π].
    """


    # Compute angular means and bootstrap error bars
    bin_centers, angular_means, angular_errors = angular_mean_with_error(prediction, target, n_bins, n_bootstrap)

    # Setup figure and plot
    for bin_center, mean, error in zip(bin_centers, angular_means, angular_errors):
        lower = mean - error
        upper = mean + error

        ax.plot([bin_center, bin_center], [lower, upper], color='gray', linestyle='-',
                alpha=0.6)  # Regular error bar
        ax.plot(bin_center, mean, 'o', color='blue')

    ax.set_xlim(0, np.pi)
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Model Kappa')
    ax.set_title('Mean with Bootstrap Error Bars')

    return


def make_error_plots(positions, abs_error, true_abs_error, kappa, mask):
    """
    Create a figure with the prediction error vs true prediction error, model kappa values, and mean kappa for
    prediction error bins.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    positions = positions[mask]

    # scatter plot of prediction error vs true prediction error
    axes[0].scatter(positions[:, 0], positions[:, 1], s=10, c=true_abs_error[mask], cmap='copper_r')
    axes[0].set_title(r"True Prediction Error")

    # kappa values
    axes[1].scatter(positions[:, 0], positions[:, 1], s=10, c=kappa[mask], cmap='copper_r')
    axes[1].set_title(r"Model Von Mises $\kappa$ Values")

    # scatter plot of kappa vs prediction error
    error_means(axes[2], abs_error, kappa, n_bins=20, n_bootstrap=100)
    axes[2].set_title("Mean Kappa for Prediction Error Bins")

    error_means(axes[3], true_abs_error, kappa, n_bins=20, n_bootstrap=100)
    axes[3].set_title("Mean Kappa for True Prediction Error Bins")

    return fig


def create_maps(positions, targets, targets_full, mu, kappa, root_dir, file_name_prefix=None, map_type=None,
                          mask_edges=None):
    """
    Create prediction and error maps for a masked subset of the galaxy distribution.
    """

    if mask_edges is None:
        mask_edges = (940, 960, 960, 980)

    mask = (
        (positions[:, 0] > mask_edges[0])
        & (positions[:, 0] < mask_edges[1])
        & (positions[:, 1] > mask_edges[2])
        & (positions[:, 1] < mask_edges[3])
    )

    if map_type is None:
        map_type = "x_component"

    if map_type == 'x_component':
        # use the inbuilt map
        targets_map, mu_map, targets_full_map = map(lambda x: np.cos(x), [targets, mu, targets_full])
        cmap = 'bwr'
    elif map_type == 'y_component':
        targets_map, mu_map, targets_full_map = map(lambda x: np.sin(x), [targets, mu, targets_full])
        cmap = 'bwr'
    elif map_type == 'angle':
        targets_map, mu_map, targets_full_map = targets, mu, targets_full
        cmap = 'twilight'
    else:
        raise ValueError(f"Map type {map_type} not recognized.")

    fig1 = make_prediction_maps(positions[mask], mu_map[mask], targets_map[mask], targets_full_map[mask], cmap)

    # Save plot
    if file_name_prefix is None:
        file_name = "prediction_map.png"
    else:
        file_name = f"{file_name_prefix}_prediction_map.png"

    save_plot(fig1, root_dir=root_dir, file_name=file_name)

    # compute abs error, always angle error
    abs_error = angular_differences(targets, mu)
    true_abs_error = angular_differences(targets_full, mu)

    fig2 = make_error_plots(positions, abs_error, true_abs_error, kappa, mask)

    # Save plot
    if file_name_prefix is None:
        file_name = "error_map.png"
    else:
        file_name = f"{file_name_prefix}_error_map.png"

    save_plot(fig2, root_dir=root_dir, file_name=file_name)

    return fig1, fig2



if __name__ == '__main__':

    # Run this script to create prediction maps from a saved model output directory

    # argparse for model path
    parser = argparse.ArgumentParser(description='Create prediction map')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--mask_edges', type=str, default='940,960,960,980')
    parser.add_argument('--map_type', type=str, default='x_component')

    args = parser.parse_args()

    # config
    config = torch.load(os.path.join(args.output_dir, 'config.pth'))

    # replace all with numpy
    predictions_mu = np.load(os.path.join(args.output_dir, 'predictions_mu.npy'))
    predictions_kappa = np.load(os.path.join(args.output_dir, 'predictions_kappa.npy'))
    targets = np.load(os.path.join(args.output_dir, 'targets.npy'))
    targets_full = np.load(os.path.join(args.output_dir, 'targets_full.npy'))
    positions = np.load(os.path.join(args.output_dir, 'positions.npy'))

    # create prediction map

    mask_edges = list(map(int, args.mask_edges.split(',')))

    create_maps(positions, targets, targets_full, predictions_mu, predictions_kappa, args.output_dir,
                          file_name_prefix=None, mask_edges=mask_edges, map_type=args.map_type)

