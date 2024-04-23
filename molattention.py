import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import GATConv, Linear
from torch_geometric.typing import Adj


class MolAttention(torch.nn.Module):
    """This is an attention-based model to use multiple types of nodes,
    to represent features and/or functional groups.

    Args:
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_heads (int): Number of attention heads.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)
    """

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.lin1 = Linear(-1, hidden_channels)

        self.gats = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            gat = GATConv(
                hidden_channels,
                hidden_channels,
                num_heads,
                add_self_loops=False,
                dropout=dropout,
                edge_dim=edge_dim,
                negative_slope=0.01,
            )
            self.gats.append(gat)

        self.lin2 = Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        for gat in self.gats:
            gat.reset_parameters()
        self.lin2.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Tensor,
    ) -> Tensor:
        """"""

        x = F.leaky_relu_(self.lin1(x))

        for gat in self.gats:
            h = gat(x, edge_index)
            h = F.elu_(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.lin2(out)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"out_channels={self.out_channels}, "
            f"num_layers={self.num_layers}, "
            f")"
        )
