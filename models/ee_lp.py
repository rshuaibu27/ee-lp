
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class SASConv(MessagePassing):
    def __init__(self, hidden_dim: int):
        super().__init__(aggr="add")
        self.Omega = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.W_s_raw = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.Omega)
        nn.init.orthogonal_(self.W_s_raw)

    @property
    def Omega_as(self):
        return self.Omega - self.Omega.t()

    @property
    def W_s(self):
        return (self.W_s_raw + self.W_s_raw.t()) / 2.0

    def forward(self, H, edge_index, num_nodes):
        edge_index_no_sl, _ = add_self_loops(edge_index, num_nodes=num_nodes,
                                              fill_value=0.0)
        row, col = edge_index_no_sl
        deg = degree(col, num_nodes, dtype=H.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        anti_term = -F.relu(H @ self.Omega_as)
        sym_term = self.propagate(edge_index_no_sl, x=H @ self.W_s,
                                  edge_weight=norm, size=(num_nodes, num_nodes))
        delta_H = F.relu(torch.tanh(anti_term + sym_term))
        return delta_H

    def message(self, x_j, edge_weight):
        return edge_weight.unsqueeze(-1) * x_j


class PairConfidenceNet(nn.Module):
    def __init__(self, hidden_dim: int, lf: int = 1):
        super().__init__()
        layers = []
        in_dim = 2 * hidden_dim
        for _ in range(lf):
            layers += [nn.Linear(in_dim, in_dim, bias=False), nn.ReLU()]
        layers += [nn.Linear(in_dim, 2, bias=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, h_u, h_v):
        return self.net(torch.cat([h_u, h_v], dim=-1))


class PairTemperatureNet(nn.Module):
    def __init__(self, hidden_dim: int, nu0: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.nu0 = nu0

    def forward(self, h_u, h_v):
        g = self.linear(torch.cat([h_u, h_v], dim=-1))
        return F.softplus(g) + self.nu0


def gumbel_softmax_sample(logits, temperature, hard=False):
    gumbels = -torch.empty_like(logits).exponential_().log()
    y = (logits + gumbels) / temperature
    y_soft = y.softmax(dim=-1)

    if hard or not logits.requires_grad:
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        return (y_hard - y_soft).detach() + y_soft
    return y_soft


class EELP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, L: int = 20,
                 lf: int = 1, nu0: float = 1.0):
        super().__init__()
        self.L = L
        self.hidden_dim = hidden_dim

        self.encoder = nn.Linear(in_dim, hidden_dim, bias=False)
        self.sas_conv = SASConv(hidden_dim)
        self.fc = PairConfidenceNet(hidden_dim, lf=lf)
        self.fv = PairTemperatureNet(hidden_dim, nu0=nu0)
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, edge_index, pairs):
        num_nodes = x.size(0)
        src, dst = pairs[:, 0], pairs[:, 1]

        H = F.relu(self.encoder(x))
        r = torch.cat([H[src], H[dst]], dim=-1)
        time_spent = torch.zeros(pairs.size(0), device=x.device)

        for l in range(self.L):
            delta_H = self.sas_conv(H, edge_index, num_nodes)

            h_u = H[src]
            h_v = H[dst]
            logits = self.fc(h_u, h_v)
            nu     = self.fv(h_u, h_v)
            c      = gumbel_softmax_sample(logits, nu,
                                           hard=not self.training)
            tau    = c[:, 0].unsqueeze(-1)

            delta_r = torch.cat([delta_H[src], delta_H[dst]], dim=-1)
            r = r + tau * delta_r
            H = H + delta_H

            time_spent = time_spent + tau.squeeze(-1).detach()

        scores = self.predictor(r).squeeze(-1)
        return scores, time_spent


class GCNLinkPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, L: int = 3):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.convs = nn.ModuleList(
            [GCNConv(in_dim if i == 0 else hidden_dim, hidden_dim)
             for i in range(L)]
        )
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, pairs):
        H = x
        for conv in self.convs:
            H = F.relu(conv(H, edge_index))
        src, dst = pairs[:, 0], pairs[:, 1]
        r = torch.cat([H[src], H[dst]], dim=-1)
        scores = self.predictor(r).squeeze(-1)
        return scores, None

class SAGELinkPredictor(nn.Module):
    """GraphSAGE baseline for comparison."""

    def __init__(self, in_dim: int, hidden_dim: int, L: int = 3):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.convs = nn.ModuleList(
            [SAGEConv(in_dim if i == 0 else hidden_dim, hidden_dim)
             for i in range(L)]
        )
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, pairs):
        H = x
        for conv in self.convs:
            H = F.relu(conv(H, edge_index))
        src, dst = pairs[:, 0], pairs[:, 1]
        r = torch.cat([H[src], H[dst]], dim=-1)
        return self.predictor(r).squeeze(-1), None


class GATLinkPredictor(nn.Module):
    """GAT baseline for comparison."""

    def __init__(self, in_dim: int, hidden_dim: int, L: int = 3,
                 heads: int = 4):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.convs = nn.ModuleList()
        for i in range(L):
            in_c  = in_dim if i == 0 else hidden_dim
            out_c = hidden_dim // heads
            self.convs.append(GATConv(in_c, out_c, heads=heads,
                                      concat=True, dropout=0.0))
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, pairs):
        H = x
        for conv in self.convs:
            H = F.relu(conv(H, edge_index))
        src, dst = pairs[:, 0], pairs[:, 1]
        r = torch.cat([H[src], H[dst]], dim=-1)
        return self.predictor(r).squeeze(-1), None