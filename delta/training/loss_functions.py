import torch
import torch.nn.functional as F
from ..utils.utils import angle_from_trig

def egnn_loss(model, model_input, targets):
    _, _, predictions = model(*model_input)
    return F.mse_loss(predictions, targets)


def vmdn_loss(model, model_inputs, targets):
    targets = angle_from_trig(targets[:, 0], targets[:, 1])
    return model.loss(*model_inputs, target=targets)

