import torch
from .utils import angular_differences, angular_mean_with_error
import matplotlib.pyplot as plt
import argparse
from .plotting import save_plot
import numpy as np
import os
import yaml
from ..data.dataloading import create_dataloaders


def make_prediction_maps(positions, predictions, targets, targets_full, cmap):
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





def make_error_plots(positions, abs_error, kappa, mask):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    positions = positions[mask]
    # kappa values
    axes[0].scatter(positions[:, 0], positions[:, 1], s=10, c=kappa[mask], cmap='copper_r')
    axes[0].set_title(r"Model Von Mises $\kappa$ Values")

    error_heatmap(axes[1], abs_error, kappa, x_variable='Prediction Error', y_variable='Model Kappa', bins=50,)
    axes[1].set_title("Kappa vs Prediction Error Heatmap")

    # scatter plot of kappa vs prediction error
    error_means(axes[2], abs_error, kappa, n_bins=20, n_bootstrap=100)
    axes[2].set_title("Mean Kappa for Prediction Error Bins")

    return fig


def create_maps(positions, targets, targets_full, mu, kappa, root_dir, file_name_prefix=None, map_type=None,
                          mask_edges=None):

    if mask_edges is None:
        mask_edges = (520, 540, 20, 40)

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
        targets, mu, targets_full = map(lambda x: np.cos(x), [targets, mu, targets_full])
        cmap = 'bwr'
    elif map_type == 'y_component':
        targets, mu, targets_full = map(lambda x: np.sin(x), [targets, mu, targets_full])
        cmap = 'bwr'
    elif map_type == 'angle':
        cmap = 'twilight'
    else:
        raise ValueError(f"Map type {map_type} not recognized.")

    fig1 = make_prediction_maps(positions[mask], mu[mask], targets[mask], targets_full[mask], cmap)

    # Save plot
    if file_name_prefix is None:
        file_name = "prediction_map.png"
    else:
        file_name = f"{file_name_prefix}_prediction_map.png"

    save_plot(fig1, root_dir=root_dir, file_name=file_name)

    # compute abs error depending on map type
    if map_type == 'x_component' or map_type == 'y_component':
        abs_error = np.abs(targets - mu)
    elif map_type == 'angle':
        abs_error = angular_differences(targets, mu)
    else:
        raise ValueError(f"Map type {map_type} not recognized.")

    fig2 = make_error_plots(positions, abs_error, kappa, mask)

    # Save plot
    if file_name_prefix is None:
        file_name = "error_map.png"
    else:
        file_name = f"{file_name_prefix}_error_map.png"

    save_plot(fig2, root_dir=root_dir, file_name=file_name)

    return fig1, fig2



if __name__ == '__main__':

    # argparse for model path
    parser = argparse.ArgumentParser(description='Create prediction map')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--mask_edges', type=str, default='520,540,20,40')
    parser.add_argument('--map_type', type=str, default='x_component')

    args = parser.parse_args()

    # config
    config = torch.load(os.path.join(args.output_dir, 'config.pth'))

    # load model
    predictions_mu = torch.load(os.path.join(args.output_dir, 'predictions_mu.pth'))
    predictions_kappa = torch.load(os.path.join(args.output_dir, 'predictions_kappa.pth'))
    targets = torch.load(os.path.join(args.output_dir, 'targets.pth'))
    targets_full = torch.load(os.path.join(args.output_dir, 'targets_true.pth'))
    positions = torch.load(os.path.join(args.output_dir, 'positions.pth'))

    # create prediction map

    mask_edges = list(map(int, args.mask_edges.split(',')))

    create_maps(positions, targets, targets_full, predictions_mu, predictions_kappa, args.output_dir,
                          file_name_prefix=None, mask_edges=mask_edges, map_type=args.map_type)

