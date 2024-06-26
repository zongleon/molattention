import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import GATConv, global_add_pool, Linear


class MolAttention(torch.nn.Module):
    """A modified Grpah Attention model based on:
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`.

    Args:
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.lin1 = Linear(-1, hidden_channels)

        self.gat_conv = GATConv(
            hidden_channels,
            hidden_channels,
            dropout=dropout,
            add_self_loops=False,
            negative_slope=0.01,
        )

        self.atom_gats = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(
                hidden_channels,
                hidden_channels,
                dropout=dropout,
                add_self_loops=False,
                negative_slope=0.01,
            )
            self.atom_gats.append(conv)

        self.mol_conv = GATConv(
            hidden_channels,
            hidden_channels,
            dropout=dropout,
            add_self_loops=False,
            negative_slope=0.01,
        )
        self.mol_conv.explain = False  # Cannot explain global pooling.

        self.lin2 = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.gat_conv.reset_parameters()
        for gat in self.atom_gats:
            gat.reset_parameters()
        self.mol_conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """"""
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.gat_conv(x, edge_index))
        x = F.dropout(h, p=self.dropout, training=self.training)

        for gat in self.atom_gats:
            h = gat(x, edge_index)
            h = F.elu(h)
            x = F.dropout(h, p=self.dropout, training=self.training)

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()

        h = F.elu_(self.mol_conv((x, out), edge_index))
        out = F.dropout(h, p=self.dropout, training=self.training)

        # Predictor:
        return self.lin2(out)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"out_channels={self.out_channels}, "
            f"num_layers={self.num_layers}, "
            f")"
        )
