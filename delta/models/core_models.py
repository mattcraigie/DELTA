import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=(), activation=nn.ReLU):
        super(MLP, self).__init__()

        layers = []

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class CompressionNetwork(nn.Module):
    def __init__(self, model):
        super(CompressionNetwork, self).__init__()
        self.model = model

    def forward(self, *args):
        h, _, v = self.model(*args)

        angle = torch.arctan2(v[:, 1], v[:, 0] + 1e-5)[:, None]

        return torch.cat([angle, h], dim=1)
