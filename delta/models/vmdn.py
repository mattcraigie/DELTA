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
    vmdn_regularization = model_config["vmdn_regularization"]
    vmdn_dropout = model_config["vmdn_dropout"]

    egnn = EGNN(num_properties, egnn_num_layers, egnn_hidden_dim)
    compression_model = CompressionNetwork(egnn)
    model = VMDN(compression_model, vmdn_hidden_layers, vmdn_regularization, vmdn_dropout)

    return model


class VMDN(nn.Module):
    """
    Von Mises Density Network (VMDN) model.
    Custom distribution network to model a Von Mises distribution.
    The model predicts the mean (mu) and concentration (kappa).
    """

    def __init__(self, compression_network, hidden_layers=None, lambda_kappa=0.0, dropout=0.0):
        super().__init__()
        self.compression_network = compression_network
        self.lambda_kappa = lambda_kappa

        if hidden_layers is None:
            hidden_layers = [32, 32]

        self.angle_network = MLP(input_dim=compression_network.out_size, output_dim=1, hidden_layers=hidden_layers)
        self.kappa_network = MLP(input_dim=compression_network.out_size, output_dim=1, hidden_layers=hidden_layers)

        self.dropout = dropout

        self.device = None


    def forward(self, *args):
        compressed = self.compression_network(*args)
        mu = (self.angle_network(compressed) % (np.pi * 2))
        log_kappa = self.kappa_network(compressed)
        log_kappa = torch.clamp(log_kappa, min=-10, max=3)  # Prevent extreme values
        kappa = torch.exp(log_kappa)
        return mu, kappa

    def loss(self, *args, target=None):
        mu, kappa = self.forward(*args)
        dist_vonmises = VonMises(mu, kappa)
        log_prob = dist_vonmises.log_prob(target)

        # Apply a random mask to introduce stochasticity
        if self.training:
            node_mask = torch.rand(log_prob.size(0)) > self.dropout  # 20% dropout rate
        else:
            node_mask = torch.ones(log_prob.size(0))
        masked_log_prob = log_prob[node_mask]
        masked_target = target[node_mask]  # Optionally mask the target too, for consistency
        masked_mu = mu[node_mask]


        # Negative log-likelihood (NLL) loss on the masked subset
        nll = -masked_log_prob.mean()

        total_loss = nll

        # Penalize tight kappa if mu is far from target, and loose kappa if mu is close to target
        if self.training:
            mu_error = torch.abs(masked_target - masked_mu) % (2 * np.pi)  # Circular distance
            tight_penalty = mu_error * kappa[node_mask]  # Penalize tightness when error is high
            loose_penalty = (2 * np.pi - mu_error) / (kappa[node_mask] + 1e-6)  # Penalize looseness when error is low

            kappa_penalty = torch.mean(tight_penalty + loose_penalty)
            total_loss += self.lambda_kappa * kappa_penalty

        return total_loss

    def sample(self, *args, n_samples=1):
        mu, kappa = self.forward(*args)
        dist_vonmises = VonMises(mu, kappa)
        samples = dist_vonmises.sample((n_samples,))
        return samples[0]

    def to(self, device):
        super().to(device)
        self.device = device
        return self