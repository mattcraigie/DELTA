import torch
import torch.nn.functional as F


def egnn_loss(model, model_input, targets):
    _, _, predictions = model(*model_input)
    return F.mse_loss(predictions, targets)


def vmdn_loss(model, model_inputs, targets):
    print("loss_functions.py: vmdn_loss: model_inputs: ", model_inputs)
    return model.loss(model_inputs, target=targets)

