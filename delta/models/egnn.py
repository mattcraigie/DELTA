import torch
import torch.nn as nn
from .basic_models import MLP
from torch_scatter import scatter
import torch.nn.functional as F

# todo: make it so there's a spin2 or non-spin2 option


class EGNN(nn.Module):
    """
    Equivariant Graph Neural Network (EGNN) model. This model is designed to predict output vectors of a system
    based on the input node properties and relative positions of the nodes. The output vectors are equivariant to the
    input positions, meaning that the output vectors will rotate with the input positions.

    Inspired by and developed by following the methodology outlined in https://arxiv.org/pdf/2102.09844
    """
    def __init__(self, num_properties, num_layers, hidden_dim):
        super().__init__()

        assert num_layers > 0, 'Number of layers must be greater than 0.'

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.node_embedding = nn.Linear(num_properties, hidden_dim)
        self.edge_mlp = MLP(hidden_dim * 2 + 1, hidden_dim, hidden_layers=[hidden_dim, ])
        self.node_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_layers=[hidden_dim,])
        self.weighting_mlp = MLP(hidden_dim, hidden_dim, hidden_layers=[hidden_dim,])

    def forward(self, h, x, edge_index):
        row, col = edge_index
        h = self.node_embedding(h)
        rel_pos = x[row] - x[col]  # 3D relative position
        rel_dist = (rel_pos ** 2).sum(dim=-1, keepdim=True) ** 0.5

        # Update positions based on relative positions
        rel_theta = torch.arctan2(rel_pos[:, 1], rel_pos[:, 0])
        rel_phi = torch.arctan2(rel_pos[:, 2], torch.sqrt(rel_pos[:, 0] ** 2 + rel_pos[:, 1] ** 2))
        rel_pos_spin2 = rel_dist * torch.stack([torch.cos(2 * rel_theta), torch.sin(2 * rel_theta), rel_phi], dim=1)

        for layer_num in range(self.num_layers):

            # calculate edge features based on node properties and relative distance to node
            edge_inputs = torch.cat([h[row], h[col], rel_dist], dim=-1)
            edge_features_ij = self.edge_mlp(edge_inputs)  # turn these values into hidden_dim edge features

            agg_edge_features_i = scatter(edge_features_ij, row, dim=0, dim_size=h.size(0), reduce='sum')  # sum the edge features for each node

            # update the node values based on the aggregated edge features
            node_inputs = torch.cat([h, agg_edge_features_i], dim=-1)
            h = h + self.node_mlp(node_inputs)  # add to existing features rather than overwriting for stability

        # calculate the vectors based on the edge features weighted by the relative positions.
        # why am I not also using node features? could also include h[row] and h[col] as well as edge_features_ij
        # I suppose edge features use those as inputs already so might be redundant
        rel_pos_scaled = rel_pos_spin2.unsqueeze(-2) * self.weighting_mlp(edge_features_ij).unsqueeze(-1)  # shape (N * K, hidden_dim, 3)
        v_latent = scatter(rel_pos_scaled, row, dim=0,
                    dim_size=x.size(0), reduce='mean') # shape (N, hidden_dim, 3)  |  This does scatter over first dim

        # use the first hidden_dim values to represent the output vector
        v = v_latent[:, 0, :2]

        # Constrain predictions to lie on the unit circle
        v = F.normalize(v, p=2, dim=-1)

        return v_latent, x, v

