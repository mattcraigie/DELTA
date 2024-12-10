import torch
from .utils import get_model_predictions
import matplotlib.pyplot as plt
import argparse
from .plotting import save_plot
import numpy as np


def load_pretrained_model(model_path):
    model = torch.load(model_path)
    return model

def create_prediction_map(model, dataloader, dataloader_full, device, root_dir, file_name_prefix, map_type=None, mask_edges=None):
    predictions, targets = get_model_predictions(model, dataloader, device)
    _, targets_full = get_model_predictions(model, dataloader_full, device)

    positions = dataloader.dataset.positions

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

    print(positions.shape)

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
    file_name = file_name_prefix + "_prediction_map.png"

    return save_plot(fig, root_dir=root_dir, file_name=file_name)


if __name__ == '__main__':

    # argparse for model path
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--data_full_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    # load model
    model = load_pretrained_model(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # create prediction map
    create_prediction_map(model, args.data_path, args.data_full_path, device, args.output_dir, map_type='angle')


