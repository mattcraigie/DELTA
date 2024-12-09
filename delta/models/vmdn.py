from torch.distributions import VonMises
import numpy as np
import torch
from torch import nn
from .egnn import EGNN
from .basic_models import MLP, CompressionNetwork

def init_vmdn(model_config):

    num_properties = model_config["num_properties"]
    egnn_num_layers = model_config["egnn_num_layers"]
    egnn_hidden_dim = model_config["egnn_hidden_dim"]
    vmdn_hidden_layers = model_config["vmdn_hidden_layers"]

    egnn = EGNN(num_properties, egnn_num_layers, egnn_hidden_dim)
    compression_model = CompressionNetwork(egnn)
    model = VMDN(compression_model, vmdn_hidden_layers)

    return model


class VMDN(nn.Module):
    """
    Von Mises Density Network (VMDN) model.
    Custom distribution network to model a Von Mises distribution.
    The model predicts the mean (mu) and concentration (kappa).
    """

    def __init__(self, dim_in, compression_network, hidden_layers=None):
        super().__init__()
        self.compression_network = compression_network

        if hidden_layers is None:
            hidden_layers = [32, 32]

        self.angle_network = MLP(input_dim=dim_in, output_dim=1, hidden_layers=hidden_layers)
        self.kappa_network = MLP(input_dim=dim_in, output_dim=1, hidden_layers=hidden_layers)


    def forward(self, *args):
        compressed = self.compression_network(*args)
        mu = (self.angle_network(compressed) % (np.pi * 2)) - np.pi
        log_kappa = self.kappa_network(compressed)
        log_kappa = torch.clamp(log_kappa, min=-3, max=3)  # Prevent extreme values
        kappa = torch.exp(log_kappa)
        return mu, kappa

    def loss(self, *args, target=None):
        mu, kappa = self.forward(*args)
        dist_vonmises = VonMises(mu, kappa)
        log_prob = dist_vonmises.log_prob(target)
        nll = -log_prob.mean()
        return nll

    def sample(self, *args, n_samples=1):
        mu, kappa = self.forward(*args)
        dist_vonmises = VonMises(mu, kappa)
        samples = dist_vonmises.sample((n_samples,))
        return samples[0]
