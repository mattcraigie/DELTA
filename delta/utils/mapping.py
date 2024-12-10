import torch
from .utils import get_model_predictions
import matplotlib.pyplot as plt
import argparse
from .plotting import save_plot


def load_pretrained_model(model_path):
    model = torch.load(model_path)
    return model

def create_prediction_map(model, dataloader, dataloader_full, device, root_dir, file_name_prefix):
    predictions, targets = get_model_predictions(model, dataloader, device)
    _, targets_full = get_model_predictions(model, dataloader_full, device)

    positions = dataloader.dataset.positions

    # mask = (
    #     (positions[:, 0] > 0)
    #     & (positions[:, 0] < 100)
    #     & (positions[:, 1] > 0)
    #     & (positions[:, 1] < 100)
    # )

    mask = slice(None)

    positions = positions[mask]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # training targets
    axs[0].scatter(positions[:, 0], positions[:, 1], c=targets[mask], s=1, cmap='Blues')
    axs[0].set_title("Training Targets")

    # model predictions
    axs[1].scatter(positions[:, 0], positions[:, 1], c=predictions[mask], s=1, cmap='Blues')
    axs[1].set_title("Model Predictions")

    # true targets
    axs[2].scatter(positions[:, 0], positions[:, 1], c=targets_full[mask], s=1, cmap='Blues')
    axs[2].set_title("True Targets")

    # Save plot
    file_name = file_name_prefix + "prediction_map.png"

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


