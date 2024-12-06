from torch.distributions import VonMises
from .core_models import MLP
import numpy as np
import torch

class VonMisesDensityNetwork(nn.Module):
    """
    Custom distribution network to model a Von Mises distribution.
    The model predicts the cosine and sine of the mean (mu) and concentration (kappa) for each dimension.
    Uses a more stable parameterization by predicting cos(mu) and sin(mu) separately.
    """

    def __init__(self, dim_in, dim_out, compression_network):
        super().__init__()
        self.compression_network = compression_network

        # Network to predict cos(mu) and sin(mu)
        self.angle_network = MLP(dim_in, dim_out, hidden_layers=[32, ])
        self.kappa_network = MLP(dim_in, dim_out, hidden_layers=[32, ])

        self.dim_out = dim_out

    def _compute_mu_from_components(self, cos_mu, sin_mu):
        """
        Safely compute mu from its trigonometric components.
        Uses atan2 which handles quadrant selection automatically.
        """
        # Normalize the components to ensure they form a valid unit vector
        norm = torch.sqrt(cos_mu ** 2 + sin_mu ** 2).clamp(min=1e-6)
        cos_mu_normalized = cos_mu / norm
        sin_mu_normalized = sin_mu / norm

        # Use atan2 for stable angle computation
        mu = torch.atan2(sin_mu_normalized, cos_mu_normalized)
        return mu

    def forward(self, *args):
        # print([a.shape for a in args])
        compressed = self.compression_network(*args)

        # continuous component approach:
        # angle_outputs = self.angle_network(compressed)
        # cos_mu, sin_mu = torch.chunk(angle_outputs, 2, dim=-1)
        # mu = self._compute_mu_from_components(cos_mu, sin_mu)

        # looping angle approach:
        mu = self.angle_network(compressed) % (np.pi * 2)

        # Handle kappa with numerical stability
        log_kappa = self.kappa_network(compressed)
        log_kappa = torch.clamp(log_kappa, min=-3, max=3)  # Prevent extreme values
        kappa = torch.exp(log_kappa)
        # kappa = torch.clamp(kappa, min=0.5)
        return mu, kappa

    def loss(self, *args, target=None):
        mu, kappa = self.forward(*args)
        # print(target[:5])

        # Create Von Mises distribution with safety checks
        valid_kappa = torch.isfinite(kappa) & (kappa > 0)
        if not valid_kappa.all():
            # Handle invalid kappa values by setting them to a small positive value
            kappa = torch.where(valid_kappa, kappa, torch.ones_like(kappa) * 1e-6)

        dist_vonmises = VonMises(mu, kappa)

        # Compute negative log likelihood with safety checks
        log_prob = dist_vonmises.log_prob(target)
        valid_probs = torch.isfinite(log_prob)
        if not valid_probs.all():
            # Handle invalid probabilities by masking them out
            log_prob = torch.where(valid_probs, log_prob,
                                   torch.zeros_like(log_prob))

        nll = -log_prob.mean()

        # Add regularization to encourage valid predictions
        # angle_reg = torch.mean((cos_mu**2 + sin_mu**2 - 1)**2)
        total_loss = nll  # + 0.01 * angle_reg

        return total_loss

    def sample(self, *args, n_samples=1):
        mu, kappa = self.forward(*args)

        # Ensure kappa is valid before sampling
        kappa = torch.clamp(kappa, min=1e-6)

        dist_vonmises = VonMises(mu, kappa)
        samples = dist_vonmises.sample((n_samples,))
        return samples[0]  # Shape: (n_samples, batch_size, n_components)
