import torch
import torch.nn.functional as F
from ..utils.utils import angle_from_trig
import numpy as np

def egnn_loss(model, model_input, targets):
    _, _, predictions = model(*model_input)
    return F.mse_loss(predictions, targets)


def vmdn_loss(model, model_inputs, targets):
    targets = torch.atan2(targets[:, 1], targets[:, 0]).unsqueeze(-1) + np.pi
    return model.loss(*model_inputs, target=targets)

