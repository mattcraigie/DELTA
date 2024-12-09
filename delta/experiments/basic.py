import torch
from ..data.dataloading import create_dataloaders
from ..models.egnn import EGNN
from ..models.vmdn import VMDN, init_vmdn
from ..models.basic_models import CompressionNetwork
from ..utils.plotting import plot_results
from ..training.train import train_model
from ..utils.utils import get_model_predictions
import os


def run_basic_experiment(config):
    """
    Run a basic test of the model.
    """

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data
    data_dir = config["data"]["data_root"]
    alignment_strength = config["data"]["alignment_strength"]
    num_neighbors = config["data"]["num_neighbors"]
    datasets, dataloaders = create_dataloaders(data_dir, alignment_strength, num_neighbors)

    # Initialize the model
    model = init_vmdn(config["model"])
    model.to(device)

    if config["model"]["pretrain"]:
        # Pre-Train the EGNN model
        pretrain_epochs = config["model"]["pretrain_epochs"]
        pretrain_learning_rate = config["model"]["pretrain_learning_rate"]
        train_model(model.compression_network.egnn, dataloaders['train'], dataloaders['val'], pretrain_epochs,
                    pretrain_learning_rate, device)

    # Train the model
    train_epochs = config["training"]["train_epochs"]
    train_learning_rate = config["training"]["train_learning_rate"]

    model, losses = train_model(model, dataloaders['train'], dataloaders['val'], train_epochs,
                                train_learning_rate, device)

    predictions, targets = get_model_predictions(model, dataloaders['val'], device)

    analysis_name = config["analysis"]["name"]
    output_dir = config["analysis"]["output_dir"]

    # create folder for the analysis
    analysis_dir = os.path.join(output_dir, analysis_name)
    os.makedirs(analysis_dir, exist_ok=True)

    # save the model, losses, predictions and targets into the folder
    torch.save(model.state_dict(), os.path.join(analysis_dir, "model.pth"))
    torch.save(losses, os.path.join(analysis_dir, "losses.pth"))
    torch.save(predictions, os.path.join(analysis_dir, "predictions.pth"))
    torch.save(targets, os.path.join(analysis_dir, "targets.pth"))
    torch.save(config, os.path.join(analysis_dir, "config.pth"))

    # plot the results
    plot_results(losses, predictions, targets, analysis_dir, analysis_name)

    # Repeat with the fully aligned data
    alignment_strength = 1.0
    _, dataloaders_full = create_dataloaders(data_dir, alignment_strength, num_neighbors)
    _, targets_full = get_model_predictions(model, dataloaders_full['val'], device)
    torch.save(targets_full, os.path.join(analysis_dir, "targets_full.pth"))

    plot_results(losses, predictions, targets_full, analysis_dir, analysis_name + "_full")
