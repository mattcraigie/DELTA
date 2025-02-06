import torch
import numpy as np
import copy
from .loss_functions import egnn_loss, vmdn_loss


def train_epoch(model, optimizer, loss_function, train_loader, device):
    """
    Train the model for one epoch.
    """
    model.train()
    epoch_losses = []

    for h, x, edge_index, v_target in train_loader:
        h = h.to(device)
        x = x.to(device)
        edge_index = edge_index.to(device)
        v_target = v_target.to(device)
        model_input = (h, x, edge_index)

        optimizer.zero_grad()
        loss = loss_function(model, model_input, v_target)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    return np.mean(epoch_losses)


def validate(model, loss_function, val_loader, device):
    """
    Validate the model on the validation set.
    """
    model.eval()
    val_losses = []

    with torch.no_grad():
        for h, x, edge_index, v_target in val_loader:
            h = h.to(device)
            x = x.to(device)
            edge_index = edge_index.to(device)
            v_target = v_target.to(device)
            model_input = (h, x, edge_index)
            val_loss = loss_function(model, model_input, v_target)
            val_losses.append(val_loss.item())

    return np.mean(val_losses)


def train_model(model,
                train_loader,
                val_loader,
                num_epochs,
                learning_rate,
                device,
                print_every=10,
                return_best_loss=False):
    """
    The main training loop.
    """

    if type(model).__name__ == 'EGNN':  # pretraining / compression model only
        loss_function = egnn_loss
    elif type(model).__name__ == 'VMDN':
        loss_function = vmdn_loss
    else:
        raise ValueError(f'Unknown model type: {type(model).__name__}')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_weights = None

    for epoch in range(num_epochs):

        train_loss = train_epoch(model, optimizer, loss_function, train_loader, device)
        train_losses.append(train_loss)

        val_loss = validate(model, loss_function, val_loader, device)
        val_losses.append(val_loss)

        # Save best model weights if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())

        if (epoch + 1) % print_every == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, '
                  f'Training Loss: {train_loss:.6f}, '
                  f'Validation Loss: {val_loss:.6f}, '
                  f'Best Validation Loss: {best_val_loss:.6f}')

    # Load the best model weights at the end of training
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f'Loaded best model weights with validation loss: {best_val_loss:.6f}')


    if return_best_loss:
        return model, {'train': train_losses, 'val': val_losses}, best_val_loss

    return model, {'train': train_losses, 'val': val_losses}
