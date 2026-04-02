"""Full STGCN classification model.

Stacks 10 STGCN_Units (spatial-temporal graph convolution blocks) to
classify skeleton sequences into emotion categories. Supports both
the basic TCN variant and the multi-branch STGCN++ variant.
"""

import math

import torch
import torch.nn as nn

from emo_mocap.models.base import BaseModel
from emo_mocap.models.stgcn.ST_units import STGCN_Unit
from emo_mocap.models.stgcn.adj_matrix import GraphAAGCN


class STGCN_Model(BaseModel):
    """Full ST-GCN classification model.

    Pipeline: Input -> BatchNorm1d -> 10 STGCN_Units -> Global Avg Pool
              -> Dropout -> Linear FC -> {"logits": (N, num_class)}

    Args:
        num_class: number of output classes
        edge_index: list of (child, parent) edge tuples
        num_nodes: number of joints in the skeleton
        in_channels: number of input channels (3 for xyz)
        dropout: dropout rate for the classifier head (default: 0.5)
        edge_weighting: if True, make adjacency matrix learnable (default: True)
        plusplus: if True, use multi-branch TCN (STGCN++) (default: True)
        unit_dropout: dropout within TCN_Unit_plus branches (default: 0.1)
    """

    def __init__(
        self,
        num_class,
        edge_index,
        num_nodes,
        in_channels,
        dropout=0.5,
        edge_weighting=True,
        plusplus=True,
        unit_dropout=0.1,
    ):
        super().__init__()
        graph = GraphAAGCN(edge_index, num_nodes)
        A = torch.as_tensor(graph.A).clone().float()
        self.register_buffer("A", A)
        A = (
            nn.Parameter(A, requires_grad=True)
            if edge_weighting
            else nn.Parameter(A, requires_grad=False)
        )
        self.num_class = num_class
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
        self.st_gcn_networks = nn.Sequential(
            STGCN_Unit(in_channels, 64, A, plusplus, residual=False, unit_dropout=unit_dropout),
            STGCN_Unit(64, 64, A, plusplus, unit_dropout=unit_dropout),
            STGCN_Unit(64, 64, A, plusplus, unit_dropout=unit_dropout),
            STGCN_Unit(64, 64, A, plusplus, unit_dropout=unit_dropout),
            STGCN_Unit(64, 128, A, plusplus, stride=2, unit_dropout=unit_dropout),
            STGCN_Unit(128, 128, A, plusplus, unit_dropout=unit_dropout),
            STGCN_Unit(128, 128, A, plusplus, unit_dropout=unit_dropout),
            STGCN_Unit(128, 256, A, plusplus, stride=2, unit_dropout=unit_dropout),
            STGCN_Unit(256, 256, A, plusplus, unit_dropout=unit_dropout),
            STGCN_Unit(256, 256, A, plusplus, unit_dropout=unit_dropout),
        )
        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        self.drop_out = nn.Dropout(dropout) if dropout > 0 else lambda x: x

    def forward(self, x):
        self._validate_input(x)
        N, C, T, V = x.size()
        # (N, C, T, V) -> (N, V*C, T) for BatchNorm
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        # (N, V*C, T) -> (N, C, T, V) back to spatial layout
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        x = self.st_gcn_networks(x)
        # Global average pooling over time and joints
        c_new = x.size(1)
        x = x.view(N, c_new, -1)
        x = x.mean(2)
        x = self.drop_out(x)
        x = self.fc(x)
        return {"logits": x}

    @property
    def output_dim(self):
        return self.num_class

    @classmethod
    def from_config(cls, config):
        return cls(
            num_class=config.model.num_class,
            edge_index=config.skeleton.inward_edges,
            num_nodes=config.skeleton.num_nodes,
            in_channels=config.model.in_channels,
            edge_weighting=getattr(config.model, "edge_weighting", False),
            plusplus=getattr(config.model, "plusplus", False),
            dropout=getattr(config.model, "dropout", 0.5),
            unit_dropout=getattr(config.model, "unit_dropout", 0.1),
        )


from emo_mocap.models.registry import register_model  # noqa: E402

register_model("stgcn", STGCN_Model)
