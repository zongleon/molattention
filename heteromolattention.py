import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import (
    GATConv,
    global_add_pool,
    Linear,
    HeteroConv,
    HeteroDictLinear,
)


class HeteroMolAttention(torch.nn.Module):
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
        node_types: list[str],
        edge_types: list[tuple[str, str, str]],
        hidden_channels: int,
        out_channels: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.lin1 = HeteroDictLinear(-1, hidden_channels, types=node_types)

        self.atom_gats = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = HeteroConv(
                {
                    edge: GATConv(
                        (-1, -1),
                        hidden_channels,
                        dropout=dropout,
                        heads=num_heads,
                        add_self_loops=False,
                        negative_slope=0.01,
                    )
                    for edge in edge_types
                },
                aggr="sum",
            )
            self.atom_gats.append(conv)

        self.mol_conv = GATConv(
            (-1, -1),
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
        for gat in self.atom_gats:
            gat.reset_parameters()
        self.mol_conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x_dict, edge_index_dict, batch) -> Tensor:
        """"""
        # Atom Embedding:
        x_dict = self.lin1(x_dict)
        x_dict = {key: F.leaky_relu_(x) for key, x in x_dict.items()}

        for gat in self.atom_gats:
            h_dict = gat(x_dict, edge_index_dict)
            h_dict = {key: F.elu_(x) for key, x in h_dict.items()}
            x_dict = {
                key: F.dropout(x, p=self.dropout, training=self.training)
                for key, x in h_dict.items()
            }

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x_dict["molecule"], batch).relu_()

        h = F.elu_(self.mol_conv((x_dict["molecule"], out), edge_index))
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
