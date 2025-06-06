import torch
import numpy as np


def angle_from_trig(costheta, sintheta):
    """
    Compute angle from x/y components.
    """
    theta = np.arctan2(sintheta, costheta)
    return theta


def circular_std(angles):
    """
    Compute circular standard deviation.
    """
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))
    return np.sqrt(-2 * np.log(np.sqrt(sin_mean ** 2 + cos_mean ** 2)))


def circular_mean(angles):
    """
    Compute circular mean.
    """
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    return angle_from_trig(cos_sum, sin_sum)


def angular_differences(angles_1, angles_2):
    """
    Compute shortest angular differences between two sets of angles.
    """
    diff_abs = np.abs(angles_1 - angles_2)
    diff_complement = 2 * np.pi - diff_abs
    return np.minimum(diff_abs % (2 * np.pi), diff_complement % (2 * np.pi))

def signed_to_unsigned_angle(angles):
    """
    Convert signed angles (between -pi and pi) to unsigned angles (between 0 and 2pi).
    """
    return (angles + 2 * np.pi) % (2 * np.pi)


def angular_mean_with_error(theta_1, theta_2, n_bins=20, n_bootstrap=100):
    """
    Calculate angular means and bootstrap error bars of theta_2 binned by theta_1.
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


def get_model_predictions(model, dataloader, device, egnn=False, output_angle=True):
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

            if egnn:
                _, _, samples = model(h, x, edge_index)  # egnn returns h, x, and v_pred, where v_pred is the samples
                if output_angle:
                    samples = torch.atan2(samples[:, 1], samples[:, 0]).unsqueeze(1)
                    samples = signed_to_unsigned_angle(samples)

            else:
                samples, _ = model(h, x, edge_index)  # vmdn return mu and log_sigma, where mu is the samples
                samples = signed_to_unsigned_angle(samples)
                if not output_angle:
                    samples = torch.hstack([torch.cos(samples), torch.sin(samples)])

            if output_angle:
                v_target = torch.atan2(v_target[:, 1], v_target[:, 0]).unsqueeze(1)
                v_target = signed_to_unsigned_angle(v_target)

            predictions.append(samples.cpu().numpy())
            targets.append(v_target.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    return predictions, targets


def get_egnn_latent_outputs(model, dataloader, device):
    """
    Get the model predictions for a dataset.
    """
    model.eval()
    all_latents = []

    with torch.no_grad():
        for h, x, edge_index, v_target in dataloader:
            h = h.to(device)
            x = x.to(device)
            edge_index = edge_index.to(device)
            latents, _, _ = model(h, x, edge_index)  # egnn returns h, x, and v_pred, where v_pred is the samples
            all_latents.append(latents.cpu().numpy())

    all_latents = np.concatenate(all_latents, axis=0)

    return all_latents


def get_vmdn_outputs(model, dataloader, device):
    """
    Get the output mu and kappa of a VMDN model for a dataset.
    """
    model.eval()
    mus = []
    kappas = []

    with torch.no_grad():
        for h, x, edge_index, _ in dataloader:
            h = h.to(device)
            x = x.to(device)
            edge_index = edge_index.to(device)
            mu, kappa = model(h, x, edge_index)
            mu = mu.cpu().numpy()
            kappa = kappa.cpu().numpy()
            mu = signed_to_unsigned_angle(mu)
            mus.append(mu)
            kappas.append(kappa)

    mus = np.concatenate(mus, axis=0)
    kappas = np.concatenate(kappas, axis=0)

    return mus, kappas



def get_improvement_percentage(prediction, target):
    """
    Calculate the improvement percentage in angular differences compared to the random guess baseline.
    """
    # Compute angular differences
    angular_diff = angular_differences(prediction, target)
    mean_angular_diff = angular_diff.mean()
    baseline = np.pi / 2

    # Calculate improvement percentage
    improvement_percentage = ((baseline - mean_angular_diff) / abs(baseline)) * 100

    return improvement_percentage
