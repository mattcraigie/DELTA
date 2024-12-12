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
        # self.edge_mlp = MLP(hidden_dim * 2 + 1, hidden_dim, hidden_layers=[hidden_dim,])
        # instead of MLP, we make it with a dropout layer
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )


        self.node_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_layers=[hidden_dim,])
        self.vector_mlp = MLP(hidden_dim, 2, hidden_layers=[hidden_dim,])
        self.coord_mlp = MLP(hidden_dim, 1, hidden_layers=[hidden_dim,])

    def forward(self, h, x, edge_index):
        row, col = edge_index

        h = self.node_embedding(h)

        for _ in range(self.num_layers):
            rel_pos = x[row] - x[col]
            rel_dist = (rel_pos ** 2).sum(dim=-1, keepdim=True)
            rel_theta = torch.arctan2(rel_pos[:, 1], rel_pos[:, 0])

            # this modified the edge positions to be placed according to spin-2
            rel_pos = rel_dist * torch.stack([torch.cos(2 * rel_theta), torch.sin(2 * rel_theta)], dim=1)

            edge_feat = torch.cat([h[row], h[col], rel_dist], dim=-1)
            m_ij = self.edge_mlp(edge_feat)

            m_i = scatter(m_ij, row, dim=0, dim_size=h.size(0), reduce='sum')

            node_inp = torch.cat([h, m_i], dim=-1)
            h = h + self.node_mlp(node_inp)

            # Predicted vector field (2D)
            v = self.vector_mlp(h)

            # Update positions based on relative positions
            v = v + scatter(rel_pos * self.coord_mlp(m_ij), row, dim=0,
                            dim_size=x.size(0), reduce='mean')

        # Constrain predictions to lie on the unit circle
        v = F.normalize(v, p=2, dim=-1)

        return h, x, v

