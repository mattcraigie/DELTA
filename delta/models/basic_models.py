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
    def __init__(self, egnn):
        super(CompressionNetwork, self).__init__()
        self.egnn = egnn
        self.out_size = egnn.hidden_dim + 1

    def forward(self, *args):
        v_latent, _, _ = self.egnn(*args)  # v_latent shape: (N, hidden_dim, 3)
        angle = torch.arctan2(v_latent[:, :, 1], v_latent[:, :, 0] + 1e-5) # (N, hidden_dim)
        return angle
