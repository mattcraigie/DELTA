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
import yaml


def run_basic_experiment(config):
    """
    Run a basic test of the model multiple times (num_repeats).
    Overwrite the best model if a better validation score is found.
    """

    # ----------------------
    # 0. Basic setup
    # ----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    analysis_name = config["analysis"]["name"]
    output_dir    = config["analysis"]["output_dir"]
    analysis_dir  = os.path.join(output_dir, analysis_name)
    os.makedirs(analysis_dir, exist_ok=True)

    data_dir           = config["data"]["data_root"]
    alignment_strength = config["data"]["alignment_strength"]
    num_neighbors      = config["data"]["num_neighbors"]

    # Number of times to repeat training
    # If you haven't declared `num_repeats` in your config, you can default to 1
    num_repeats = config["analysis"].get("num_repeats", 1)

    # ----------------------
    # 1. Load data & model
    # ----------------------
    datasets, dataloaders = create_dataloaders(data_dir, alignment_strength, num_neighbors)

    # Disable properties for train/val so all are set to 1.0
    datasets['train'].h = np.ones((datasets['train'].h.shape[0], 1), dtype=np.float32)
    datasets['val'].h   = np.ones((datasets['val'].h.shape[0],   1), dtype=np.float32)




    # ----------------------
    # 3. Training loop repeated num_repeats times
    # ----------------------
    train_epochs         = config["training"]["train_epochs"]
    train_learning_rate  = config["training"]["train_learning_rate"]

    best_val_score   = None
    best_model_state = None
    best_losses      = None

    for i in range(num_repeats):
        print(f"\n=== Repeat {i+1}/{num_repeats} ===")

        # Initialize model
        model = init_vmdn(config["model"])
        model.to(device)

        if config["training"]["load_pretrained_compression"]:
            pretrain_path = os.path.join(analysis_dir, "compression_model.pth")
            try:
                model.compression_network.egnn.load_state_dict(torch.load(pretrain_path))
            except FileNotFoundError:
                print(f"Pretrained model not found at {pretrain_path}. Training from scratch.")

        elif config["training"]["pretrain"]:
            # Pre-train the EGNN model
            pretrain_epochs = config["training"]["pretrain_epochs"]
            pretrain_learning_rate = config["training"]["pretrain_learning_rate"]
            train_model(
                model.compression_network.egnn,
                dataloaders['train'],
                dataloaders['val'],
                pretrain_epochs,
                pretrain_learning_rate,
                device
            )

        elif config["training"]["load_pretrained_full"]:
            pretrain_path = os.path.join(analysis_dir, "model.pth")
            try:
                model.load_state_dict(torch.load(pretrain_path))
            except FileNotFoundError:
                print(f"Pretrained model not found at {pretrain_path}. Training from scratch.")

        # Train
        model, losses, final_val_loss = train_model(
            model,
            dataloaders['train'],
            dataloaders['val'],
            train_epochs,
            train_learning_rate,
            device,
            return_best_loss=True
        )

        # Check if this is the best run so far
        if (best_val_score is None) or (final_val_loss < best_val_score):
            best_val_score   = final_val_loss
            best_model_state = {
                k: v.cpu() for k, v in model.state_dict().items()
            }  # store on CPU
            best_losses      = losses

    # ----------------------
    # 4. Load the best model, save to disk, and do the rest of the steps
    # ----------------------
    # Load best model weights:
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save pre-trained model
    if config["training"]["pretrain"]:
        torch.save(model.compression_network.egnn.state_dict(),
                   os.path.join(analysis_dir, "compression_model.pth"))

    # Save the final best model and losses

    torch.save(model.state_dict(), os.path.join(analysis_dir, "model.pth"))
    np.save(os.path.join(analysis_dir, "losses.npy"), best_losses)

    # Now do your validation predictions with the best model
    predictions, targets = get_model_predictions(model, dataloaders['val'], device)
    predictions_mu, predictions_kappa = get_vmdn_outputs(model, dataloaders['val'], device)
    positions = datasets['val'].positions

    np.save(os.path.join(analysis_dir, "predictions.npy"),       predictions)
    np.save(os.path.join(analysis_dir, "predictions_mu.npy"),    predictions_mu)
    np.save(os.path.join(analysis_dir, "predictions_kappa.npy"), predictions_kappa)
    np.save(os.path.join(analysis_dir, "targets.npy"),           targets)
    np.save(os.path.join(analysis_dir, "positions.npy"),         positions)
    np.save(os.path.join(analysis_dir, "properties.npy"),        datasets['val'].h)

    # Save config
    with open(os.path.join(analysis_dir, "config.yaml"), 'w') as file:
        yaml.dump(config, file)

    # ----------------------
    # 5. Additional steps: alignment_strength=1.0
    # ----------------------
    alignment_strength = 1.0
    dataset_full, dataloaders_full = create_dataloaders(data_dir, alignment_strength, num_neighbors)
    dataset_full['val'].h = np.ones((dataset_full['val'].h.shape[0], 1), dtype=np.float32)

    # Evaluate with fully aligned data
    _, targets_full = get_model_predictions(model, dataloaders_full['val'], device)
    np.save(os.path.join(analysis_dir, "targets_full.npy"), targets_full)

    # Generate plots and maps
    plot_results(best_losses, predictions, targets, analysis_dir, file_name_prefix='data')
    plot_results(best_losses, predictions, targets_full, analysis_dir, file_name_prefix="true")
    create_maps(positions, targets, targets_full, predictions_mu, predictions_kappa, root_dir=analysis_dir)