import matplotlib.pyplot as plt
import torch
import os
from delta.utils.utils import angular_mean_with_error, angular_differences, angle_from_trig
import numpy as np


def save_plot(fig, root_dir=None, file_name=None, subdir="plots"):
    """
    Save a matplotlib figure to the specified directory and file name.
    """
    if root_dir is None:
        root_dir = "."
    if file_name is None:
        file_name = "plot.png"

    # Create the subdirectory if it doesn't exist
    save_dir = os.path.join(root_dir, subdir)
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot
    file_path = os.path.join(save_dir, file_name)
    fig.savefig(file_path)
    plt.close(fig)  # Close the figure to free up memory

    return file_path


def plot_losses(train_losses, val_losses, logy=True, root_dir=None, file_name=None):
    """
    Plot the training and validation losses and save the figure.
    """

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    if logy:
        plt.yscale("log")

    if file_name is None:
        file_name = "losses.png"

    return save_plot(plt.gcf(), root_dir=root_dir, file_name=file_name)




def plot_predictions_heatmap(predictions, targets, x_variable='Predictions', y_variable='Targets', bins=50,
                             hist_range=None, pmax=None, cmap='Blues', root_dir=None, file_name=None):
    """
    Plot a heatmap of the model predictions and targets. The heatmap shows the conditional distribution P(y|x).
    """

    # Compute joint histogram
    joint_histogram, xedges, yedges = np.histogram2d(predictions, targets, bins=bins, range=hist_range, density=True)

    # Normalize joint histogram to form joint PDF
    joint_pdf = joint_histogram / joint_histogram.sum()

    # Conditional distribution P(y|x)
    sum_x = joint_pdf.sum(axis=1)[:, np.newaxis]
    py_given_x = joint_pdf / (sum_x + (sum_x == 0))

    # Determine maximum value for consistent color scale
    if pmax is None:
        pmax = py_given_x.max()

    # Setup figure and plot
    fig, ax = plt.subplots(figsize=(6, 6))

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    im = ax.imshow(py_given_x.T, origin='lower', extent=extent, aspect='auto', cmap=cmap, vmax=pmax)
    ax.set_title(f'Conditional P({y_variable}|{x_variable})')
    ax.set_xlabel(x_variable)
    ax.set_ylabel(y_variable)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Probability')

    if file_name is None:
        file_name = "predictions_heatmap.png"

    return save_plot(fig, root_dir=root_dir, file_name=file_name)


def plot_angular_means(prediction, target, n_bins=20, n_bootstrap=100, root_dir=None, file_name=None):
    """
    Plot angular means of target binned by prediction with bootstrap error bars,
    wrapping error bars within the range [-π, π].
    """

    def wrap_angle(angle):
        """Wrap an angle to the range [-π, π]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # Compute angular means and bootstrap error bars
    bin_centers, angular_means, angular_errors = angular_mean_with_error(prediction, target, n_bins, n_bootstrap)

    # Setup figure and plot
    fig, ax = plt.subplots(figsize=(6, 4))

    for bin_center, mean, error in zip(bin_centers, angular_means, angular_errors):
        lower = wrap_angle(mean - error)
        upper = wrap_angle(mean + error)

        if lower > upper:  # Wrap case
            ax.plot([bin_center, bin_center], [lower, np.pi], color='gray', linestyle='-', alpha=0.6)  # Wrap lower
            ax.plot([bin_center, bin_center], [-np.pi, upper], color='gray', linestyle='-', alpha=0.6)  # Wrap upper
        else:
            ax.plot([bin_center, bin_center], [lower, upper], color='gray', linestyle='-',
                    alpha=0.6)  # Regular error bar

        # Plot the mean
        ax.plot(bin_center, mean, 'o', color='blue')

    ax.plot([-np.pi, np.pi], [-np.pi, np.pi], 'k--', alpha=0.5)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel('Predicted Angle')
    ax.set_ylabel('Target Angle')
    ax.set_title('Angular Mean with Bootstrap Error Bars')

    # Save plot
    if file_name is None:
        file_name = "prediction_mean_error.png"
    file_path = save_plot(fig, root_dir=root_dir, file_name=file_name)

    return file_path


def plot_angular_differences(prediction, target, root_dir=None, file_name=None):
    """
    Plot angular differences between prediction and target as a step histogram.
    Also plot an analytic uniform distribution.

    Parameters:
        prediction (np.ndarray): Array of predicted angular values.
        target (np.ndarray): Array of target angular values.
        root_dir (str): Directory to save the plot.

    Returns:
        file_path (str): Path to the saved plot.
    """
    # Compute angular differences
    angular_diff = angular_differences(prediction, target)
    mean_angular_diff = angular_diff.mean()
    baseline = np.pi / 2
    improvement_percentage = ((baseline - mean_angular_diff) / abs(baseline)) * 100

    # Compute histogram
    n_bins = 50  # Number of bins for the histogram
    hist, bins = np.histogram(angular_diff, bins=n_bins, range=(0, np.pi), density=True)

    # Bin centers for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Setup figure and plot
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot step histogram of angular differences
    ax.step(bin_centers, hist, where='mid', label='Observed Angular Differences', color='b')

    # Plot analytic uniform distribution (density for uniform distribution)
    uniform_density = 1 / np.pi  # Analytic uniform density over [0, pi]
    ax.axhline(uniform_density, color='r', linestyle='--', label='Uniform Distribution')

    # Set plot labels, title, and legend
    ax.set_xlabel('Angular Difference (radians)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Angular Differences, Improvement: {:.2f}%'.format(improvement_percentage))
    ax.legend()

    ax.set_ylim(0, ax.get_ylim()[1])

    # Save plot
    if file_name is None:
        file_name = "angular_differences.png"

    return save_plot(fig, root_dir=root_dir, file_name=file_name)


def plot_results(losses, predictions, targets, analysis_dir, file_name_prefix=None, egnn=False):

    # file names
    if file_name_prefix is not None:
        file_name_prefix = file_name_prefix + '_'
    else:
        file_name_prefix = ''

    # Plot losses
    if losses is not None:
        file_name_losses = file_name_prefix + 'losses.png'
        plot_losses(losses['train'], losses['val'], root_dir=analysis_dir, file_name=file_name_losses)

    predictions, targets = predictions.squeeze(1), targets.squeeze(1)

    # Plot heatmap of angular predictions and targets
    file_name_heatmap = file_name_prefix + 'heatmap.png'
    plot_predictions_heatmap(predictions, targets, x_variable='Predicted Angle', y_variable='Target Angle',
                             root_dir=analysis_dir, file_name=file_name_heatmap)

    # Plot averaged angular predictions and targets
    file_name_means = file_name_prefix + 'means.png'
    plot_angular_means(predictions, targets, root_dir=analysis_dir, file_name=file_name_means)

    # Plot histogram of angular predictions and targets
    file_name_histogram = file_name_prefix + 'differences.png'
    plot_angular_differences(predictions, targets, root_dir=analysis_dir, file_name=file_name_histogram)
