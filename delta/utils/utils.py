import torch
import numpy as np


def angle_from_trig(costheta, sintheta):
    theta = np.arctan2(sintheta, costheta)
    return theta


def circular_std(angles):
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))
    return np.sqrt(-2 * np.log(np.sqrt(sin_mean ** 2 + cos_mean ** 2)))


def circular_mean(angles):
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    return angle_from_trig(cos_sum, sin_sum)


def angular_differences(angles_1, angles_2):
    mean_1 = circular_mean(angles_1)
    mean_2 = circular_mean(angles_2)
    mean_diff = angle_from_trig(np.cos(mean_1 - mean_2), np.sin(mean_1 - mean_2))
    diff_abs = np.abs(mean_diff - angles_1)
    differences = np.minimum(diff_abs, 2 * np.pi - diff_abs)
    return differences


def angular_mean_with_error(theta_1, theta_2, n_bins=20, n_bootstrap=100):
    """
    Calculate angular means and bootstrap error bars of theta_2 binned by theta_1.

    Parameters:
        theta_1 (array-like): Array of angular values for theta_1 (independent variable).
        theta_2 (array-like): Array of angular values for theta_2 (dependent variable).
        n_bins (int): Number of bins for theta_1.
        n_bootstrap (int): Number of bootstrap resamples for error estimation.

    Returns:
        bin_centers (np.ndarray): Centers of the theta_1 bins.
        angular_means (np.ndarray): Angular means of theta_2 for each bin.
        angular_errors (np.ndarray): Bootstrap error bars for the angular means.
    """
    # Ensure theta_1 and theta_2 are wrapped between -pi and pi
    theta_1 = (theta_1 + np.pi) % (2 * np.pi) - np.pi
    theta_2 = (theta_2 + np.pi) % (2 * np.pi) - np.pi

    # Define bins for theta_1
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Initialize lists for angular means and errors
    angular_means = []
    angular_errors = []

    # Compute angular mean and bootstrap error for each bin
    for i in range(len(bins) - 1):
        # Indices of theta_1 values in the current bin
        bin_mask = (theta_1 >= bins[i]) & (theta_1 < bins[i + 1])
        theta_2_in_bin = theta_2[bin_mask]

        if len(theta_2_in_bin) > 0:
            # Compute original angular mean using the circular_mean function
            mean_angle = circular_mean(theta_2_in_bin)
            angular_means.append(mean_angle)

            # Bootstrap resampling
            bootstrap_means = []
            for _ in range(n_bootstrap):
                resample = np.random.choice(theta_2_in_bin, size=len(theta_2_in_bin), replace=True)
                resample_mean_angle = circular_mean(resample)
                bootstrap_means.append(resample_mean_angle)

            # Compute standard deviation of bootstrap means using circular_std
            angular_error = circular_std(np.array(bootstrap_means))
            angular_errors.append(angular_error)
        else:
            # Handle empty bins
            angular_means.append(np.nan)
            angular_errors.append(np.nan)

    # Convert results to numpy arrays for easier handling
    angular_means = np.array(angular_means)
    angular_errors = np.array(angular_errors)

    return bin_centers, angular_means, angular_errors


def get_model_predictions(model, dataloader, device):
    """
    Get the model predictions for a dataset.
    """
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for h, x, edge_index, v_target in dataloader:
            h = h.to(device)
            x = x.to(device)
            edge_index = edge_index.to(device)
            v_target = v_target.to(device)

            _, _, v_pred = model(h, x, edge_index)
            predictions.append(v_pred.cpu().numpy())
            targets.append(v_target.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    return predictions, targets
