import torch
import torch.nn as nn
from .basic_models import MLP
from torch_scatter import scatter
import torch.nn.functional as F

# todo: make it so there's a spin2 or non-spin2 option


class EGNN(nn.Module):
    def __init__(self, num_properties, num_layers, hidden_dim):
        super().__init__()

        assert num_layers > 0, 'Number of layers must be greater than 0.'

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.node_embedding = nn.Linear(num_properties, hidden_dim)
        self.edge_mlp = MLP(hidden_dim * 2 + 1, hidden_dim, hidden_layers=[hidden_dim, ])
        self.node_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_layers=[hidden_dim,])
        # self.vector_mlp = MLP(hidden_dim, 2, hidden_layers=[hidden_dim,])
        self.first_weighting_mlp = MLP(hidden_dim, 1, hidden_layers=[hidden_dim,])
        self.subsequent_weighting_mlp = MLP(hidden_dim + 3 * hidden_dim, 1, hidden_layers=[hidden_dim,])

    def forward(self, h, x, edge_index):
        row, col = edge_index
        h = self.node_embedding(h)
        rel_pos = x[row] - x[col]  # 3D relative position
        rel_dist = (rel_pos ** 2).sum(dim=-1, keepdim=True)

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

            '''
            # Compute latent-size features for relative positions
            rel_pos_scaled = self.coord_mlp(m_ij).unsqueeze(-1) * rel_pos.unsqueeze(1)  # [num_edges, latent_size, 3]

            # Scatter operation to aggregate the scaled positions
            v = scatter(rel_pos_scaled, row, dim=0, dim_size=x.size(0), reduce='mean')  # [batch_size, latent_size, 3]

            # Extract the 2D components (x, y) of the vector field
            v = v[..., :2]  # [batch_size, latent_size, 2]
            '''

            # calculate the equivariant vectors
            if layer_num == 0:
                weighting_inputs = edge_features_ij
                rel_pos_scaled = rel_pos_spin2.unsqueeze(1) * self.first_weighting_mlp(weighting_inputs).unsqueeze(-1)
            else:
                weighting_inputs = torch.cat([edge_features_ij, v.reshape(-1, 3 * self.hidden_dim)], dim=-1)
                rel_pos_scaled = rel_pos_spin2.unsqueeze(1) * self.subsequent_weighting_mlp(weighting_inputs).unsqueeze(1)

            v = scatter(rel_pos_scaled, row, dim=0,
                        dim_size=x.size(0), reduce='mean')  # shape (N, hidden_dim, 3)


        # Constrain predictions to lie on the unit circle
        v = F.normalize(v[:, :2], p=2, dim=-1)

        return h, x, v  # todo: output only v


