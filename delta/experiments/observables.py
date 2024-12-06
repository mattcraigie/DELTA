import torch
from torch.utils.data import DataLoader
import numpy as np
from ..data.dataloading import load_dataset, split_dataset, GraphDataset, collate_fn, create_dataloaders
from ..models.egnn import EGNN
from ..utils.plotting import plot_results
from ..training.train_egnn import train_egnn_model
from ..utils.utils import get_model_predictions
import os

'''
This experiment adds in an 'informative' observable based on the properties array, and generates an 'uninformative' 
random observable. Then, the model is trained and the influence of the observables on the model's performance is
evaluated with a SHAP analysis. 

There are two sub-experiments:

1. Permutation Experiment: The model is trained as usual, then evaluated with a random permutation of each observable 
individually, destroying the relationship between the observable and the target. 

2. Ablation Experiment: The model is re-trained leaving out each observable individually. 
'''


def make_informative_observable(properties):
    """
    Create an informative observable based on the properties. It is standard normal noise + properties.
    """
    informative_property = properties[:, 0] * 2 - 1 # scale to [-1, 1]
    informative_observable = informative_property + np.random.normal(0, 1, len(informative_property))

    return informative_observable


def make_uninformative_observable(properties):
    """
    Create an uninformative observable. It is just standard normal noise.
    """
    uninformative_observable = np.random.normal(0, 1, len(properties))

    return uninformative_observable


def create_dataloaders_observables(root_dir, alignment_strength, num_neighbors):
    """
    Create data loaders for training and validation sets.
    """
    data = load_dataset(root_dir, alignment_strength)

    informative_observable = make_informative_observable(data['properties'])
    uninformative_observable = make_uninformative_observable(data['properties'])
    data['properties'] = np.column_stack((informative_observable, uninformative_observable))

    train_data, val_data = split_dataset(data)

    train_dataset = GraphDataset(train_data['positions'],
                                 train_data['orientations'],
                                 train_data['properties'],
                                 num_neighbors)

    val_dataset = GraphDataset(val_data['positions'],
                               val_data['orientations'],
                               val_data['properties'],
                               num_neighbors)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    datasets = {'train': train_dataset, 'val': val_dataset}
    dataloaders = {'train': train_loader, 'val': val_loader}

    return datasets, dataloaders


def run_observables_experiment(config):
    """
    Run an observables test of the model.
    """
    print("Running permutation experiment...")

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    analysis_name = config["analysis"]["name"]
    output_dir = config["analysis"]["output_dir"]

    data_dir = config["data"]["data_root"]
    alignment_strength = config["data"]["alignment_strength"]
    num_neighbors = config["data"]["num_neighbors"]

    num_properties = config["model"]["num_properties"]
    num_layers = config["model"]["num_layers"]
    hidden_dim = config["model"]["hidden_dim"]

    num_epochs = config["training"]["num_epochs"]
    learning_rate = config["training"]["learning_rate"]
    loss_function = config["training"]["loss_function"]

    datasets, dataloaders = create_dataloaders_observables(data_dir, alignment_strength, num_neighbors)
    model = EGNN(num_properties, num_layers, hidden_dim)
    model.to(device)
    model, losses = train_egnn_model(model, dataloaders['train'], dataloaders['val'], num_epochs, learning_rate,
                                     device,
                                     loss_function)

    predictions, targets = get_model_predictions(model, dataloaders['val'], device)

    # plot the results
    analysis_dir = os.path.join(output_dir, analysis_name)
    plot_results(losses, predictions, targets, analysis_dir, analysis_name)

    # Repeat with the fully aligned data
    _, dataloaders_full = create_dataloaders(data_dir, alignment_strength="1.0", num_neighbors=num_neighbors)
    _, targets_full = get_model_predictions(model, dataloaders_full['val'], device)

    plot_results(losses, predictions, targets_full, analysis_dir, analysis_name + "_full")

    num_columns = datasets['val'].h.shape[1]
    for i in range(num_columns):
        # Permute the observable
        observable = datasets['val'].h[:, i]
        observable_permuted = np.random.permutation(observable)
        datasets['val'].h[:, i] = observable_permuted

        # Get the predictions with the permuted data
        predictions_permuted, _ = get_model_predictions(model, dataloaders['val'], device)

        # Plot the results
        plot_results(None, predictions_permuted, targets, analysis_dir,
                     file_name_prefix=f"{analysis_name}_permuted_{i}")
        plot_results(None, predictions_permuted, targets_full, analysis_dir,
                     file_name_prefix=f"{analysis_name}_permuted_{i}_full")

        # reset the observable to its original un-permuted state
        datasets['val'].h[:, i] = observable

    print("Permutation experiment complete.")

    # Ablation Test
    print("Running ablation experiment...")

    original_train_h = datasets['train'].h.copy()
    original_val_h = datasets['val'].h.copy()

    # The ablation will reduce the number of parameters. It will re-train.
    for i in range(num_columns):
        # Reset the dataset to the original state for this iteration
        datasets['train'].h = original_train_h.copy()
        datasets['val'].h = original_val_h.copy()

        # Remove the observable
        datasets['train'].h = np.delete(datasets['train'].h, i, axis=1)
        datasets['val'].h = np.delete(datasets['val'].h, i, axis=1)

        # Re-train the model
        model = EGNN(num_properties - 1, num_layers, hidden_dim)
        model.to(device)
        model, losses = train_egnn_model(model, dataloaders['train'], dataloaders['val'], num_epochs, learning_rate,
                                         device,
                                         loss_function)


        # Get the predictions
        predictions_ablated, _ = get_model_predictions(model, dataloaders['val'], device)

        # Plot the results
        plot_results(losses, predictions_ablated, targets, analysis_dir,
                     file_name_prefix=f"{analysis_name}_ablated_{i}")
        plot_results(losses, predictions_ablated, targets_full, analysis_dir,
                     file_name_prefix=f"{analysis_name}_ablated_{i}_full")




