import torch
import torch.nn as nn
from .basic_models import MLP
from torch_scatter import scatter
import torch.nn.functional as F

# todo: make it so there's a spin2 or non-spin2 option


class EGNN(nn.Module):
    def __init__(self, num_properties, num_layers, hidden_dim):
        super().__init__()
        assert num_layers > 0

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.node_embedding = nn.Linear(num_properties, hidden_dim)

        # Edge MLP now only outputs a scalar (or a small vector).
        # That scalar = "message strength" or "edge weight".
        self.edge_mlp = MLP(hidden_dim * 2 + 1, hidden_dim, hidden_layers=[hidden_dim])

        self.node_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_layers=[hidden_dim])

        # Optional separate MLP for coordinate updates to get a scalar
        self.coord_mlp = MLP(hidden_dim, 1, hidden_layers=[hidden_dim])

    def forward(self, h, x, edge_index):
        row, col = edge_index
        h = self.node_embedding(h)

        for _ in range(self.num_layers):
            rel_pos = x[row] - x[col]  # shape: [num_edges, 3]
            rel_dist = (rel_pos ** 2).sum(dim=-1, keepdim=True)  # shape: [num_edges, 1]

            edge_feat = torch.cat([h[row], h[col], rel_dist], dim=-1)
            e_ij = self.edge_mlp(edge_feat)  # shape: [num_edges, hidden_dim]

            # Scatter to get aggregated edge feature -> node
            m_i = scatter(e_ij, row, dim=0, dim_size=h.size(0), reduce='sum')

            # Node-level update
            node_inp = torch.cat([h, m_i], dim=-1)
            h = h + self.node_mlp(node_inp)

            # ---- Position update (makes it equivariant) ----
            # Typically we reduce e_ij to a scalar for coordinate update.
            # E.g., alpha_ij = coord_mlp(e_ij) -> shape: [num_edges, 1]
            alpha_ij = self.coord_mlp(e_ij)  # shape: [num_edges, 1]

            # Weighted difference vectors
            # shape: [num_edges, 3]
            coord_msg = alpha_ij * rel_pos

            # Sum over incoming edges
            delta_x = scatter(coord_msg, row, dim=0, dim_size=x.size(0), reduce='sum')

            # Update node positions
            x = x + delta_x

        return h, x


