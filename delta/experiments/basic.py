import torch
from ..data.dataloading import create_dataloaders
from ..models.egnn import EGNN
from ..models.vmdn import VMDN, init_vmdn
from ..models.basic_models import CompressionNetwork
from ..utils.plotting import plot_results
from ..training.train import train_model
from ..utils.utils import get_model_predictions, get_vmdn_outputs
from ..utils.mapping import create_maps
import os
import numpy as np
from ..experiments.distances import run_distance_experiment


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

    # disable properties
    datasets['train'].h = np.ones((datasets['train'].h.shape[0], 1), dtype=np.float32)
    datasets['val'].h = np.ones((datasets['val'].h.shape[0], 1), dtype=np.float32)

    # Initialize the model
    model = init_vmdn(config["model"])
    model.to(device)

    if config["training"]["pretrain"]:
        # Pre-Train the EGNN model
        pretrain_epochs = config["training"]["pretrain_epochs"]
        pretrain_learning_rate = config["training"]["pretrain_learning_rate"]
        train_model(model.compression_network.egnn, dataloaders['train'], dataloaders['val'], pretrain_epochs,
                    pretrain_learning_rate, device)

    # Train the model
    train_epochs = config["training"]["train_epochs"]
    train_learning_rate = config["training"]["train_learning_rate"]

    model, losses = train_model(model, dataloaders['train'], dataloaders['val'], train_epochs,
                                train_learning_rate, device)

    predictions, targets = get_model_predictions(model, dataloaders['val'], device)

    predictions_mu, predictions_kappa = get_vmdn_outputs(model, dataloaders['val'], device)

    # positions
    positions = datasets['val'].positions

    analysis_name = config["analysis"]["name"]
    output_dir = config["analysis"]["output_dir"]

    # create folder for the analysis
    analysis_dir = os.path.join(output_dir, analysis_name)
    os.makedirs(analysis_dir, exist_ok=True)


    # todo: don't save everything as torch tensors
    # save the model, losses, predictions and targets into the folder
    # torch.save(model.state_dict(), os.path.join(analysis_dir, "model.pth"))
    # torch.save(losses, os.path.join(analysis_dir, "losses.pth"))
    # torch.save(predictions, os.path.join(analysis_dir, "predictions.pth"))
    # torch.save(predictions_mu, os.path.join(analysis_dir, "predictions_mu.pth"))
    # torch.save(predictions_kappa, os.path.join(analysis_dir, "predictions_kappa.pth"))
    # torch.save(targets, os.path.join(analysis_dir, "targets.pth"))
    # torch.save(config, os.path.join(analysis_dir, "config.pth"))
    # torch.save(positions, os.path.join(analysis_dir, "positions.pth"))

    # # Repeat with the fully aligned data
    # alignment_strength = 1.0
    # dataset_full, dataloaders_full = create_dataloaders(data_dir, alignment_strength, num_neighbors)
    #
    # # disable properties
    # dataset_full['val'].h = np.ones((dataset_full['val'].h.shape[0], 1), dtype=np.float32)
    #
    # _, targets_full = get_model_predictions(model, dataloaders_full['val'], device)
    # torch.save(targets_full, os.path.join(analysis_dir, "targets_true.pth"))
    #
    #
    # # basic test results
    # plot_results(losses, predictions, targets, analysis_dir, file_name_prefix='data')
    # plot_results(losses, predictions, targets_full, analysis_dir, file_name_prefix="true")
    # create_maps(positions, targets, targets_full, predictions_mu, predictions_kappa, analysis_dir)

    # distances test results
    run_distance_experiment(model, datasets['val'].positions, datasets['val'].orientations, datasets['val'].h, k=num_neighbors,
                            max_distance=100, device=device,
                            analysis_dir=analysis_dir, file_name_prefix='data')