import torch
from delta.data.dataloading import create_dataloaders
from delta.models.egnn import EGNN
from delta.utils.plotting import plot_results
from delta.training.train_egnn import train_egnn_model
from delta.utils.utils import get_model_predictions
import os


def run_basic_experiment(config):
    """
    Run a basic test of the model.
    """
    print("Running basic experiment...")

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data
    root_dir = config["data"]["data_root"]
    alignment_strength = config["data"]["alignment_strength"]
    num_neighbors = config["data"]["num_neighbors"]
    datasets, dataloaders = create_dataloaders(root_dir, alignment_strength, num_neighbors)

    # Initialize the model
    num_properties = config["model"]["num_properties"]
    num_layers = config["model"]["num_layers"]
    hidden_dim = config["model"]["hidden_dim"]
    model = EGNN(num_properties, num_layers, hidden_dim)

    # Train the model
    num_epochs = config["training"]["num_epochs"]
    learning_rate = config["training"]["learning_rate"]
    loss_function = config["training"]["loss_function"]
    model, losses = train_egnn_model(model, dataloaders['train'], dataloaders['val'], num_epochs, learning_rate, device,
                                loss_function)

    predictions, targets = get_model_predictions(model, dataloaders['val'], device)

    analysis_name = config["analysis"]["name"]

    # create folder for the analysis
    analysis_dir = os.path.join(root_dir, analysis_name)
    os.makedirs(root_dir, exist_ok=True)

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
    _, dataloaders_full = create_dataloaders(root_dir, alignment_strength, num_neighbors)
    _, targets_full = get_model_predictions(model, dataloaders_full['val'], device)
    torch.save(targets_full, os.path.join(analysis_dir, "targets_full.pth"))

    plot_results(losses, predictions, targets_full, analysis_dir, analysis_name + "_full")

    print("Basic experiment completed.")