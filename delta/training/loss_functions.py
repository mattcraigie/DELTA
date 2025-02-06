import torch
import torch.nn.functional as F
from ..utils.utils import signed_to_unsigned_angle
import numpy as np

def egnn_loss(model, model_input, targets):
    """
    Compute the mean squared error loss for the EGNN model.
    """
    _, _, predictions = model(*model_input)
    return F.mse_loss(predictions, targets)


def vmdn_loss(model, model_inputs, targets):
    """
    Compute the negative log likelihood loss for the VMDN model.
    """
    targets = torch.atan2(targets[:, 1], targets[:, 0]).unsqueeze(-1)
    targets = signed_to_unsigned_angle(targets)  # converts to [0, 2pi] range
    return model.loss(*model_inputs, target=targets)

