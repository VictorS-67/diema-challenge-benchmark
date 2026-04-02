"""
For the spatial convolution, there are actually multiple convolutions. Since we separated the neighboring vertices in the skeleton into multiple subsets (self, inward and outward), we will apply a different convolution for each subset. In practice, what happens is that
1. We take one the the three matrices contained in A : this matrix Ai will be of shape (V, V)
2. We multiply the data of shape (N, C\*T, V) with Ai. More precisely, Ai is broadcasted N times, and the matrix multiplication between matrices of shapes (C\*T, V) and (V, V) is done N times (called a batch matrix multiplication by pytorch). Let's call Bn a matrix of shape (C\*T, V). Bn 
3. Since Ai is (a part of) the adjacency matrix, and has 0 wherever there is no edges between vertices, the multiplication of Bn and Ai will only keep include the contribution of the vertices that are connected (according to the subset we are in at least). We then apply the Conv2d to this batch matrix multiplication, with kernel size 1. Why Kernel size 1 ? The actual message passing between nodes comes from the matrix multiplication, which aggregates the information coming from the connected vertices. The convolution will just give a learnable weight for the contribution of this pre-aggregated message to the out_channel.

In addition to the message passing, a residual link is added in the spatial module. This comes from STGCN++.
"""

import math
import torch
import torch.nn as nn
from emo_mocap.models.weights_init import conv_init, bn_init

class Basic_GCN_Unit(nn.Module):

    """
    Basic Graph Convolutional Network Unit for skeleton-based action recognition.
    
    This module performs spatial graph convolution on motion capture sequences to aggregate
    information from neighboring joints within each frame. Unlike temporal convolution that
    operates across time, this unit captures spatial relationships by processing joint
    connections defined by the skeleton's adjacency matrix.
    
    Architecture:
        The unit implements multi-subset graph convolution where neighboring joints are
        classified into multiple subsets (typically 3: self, inward, outward). Each subset
        has its own learnable convolution parameters, allowing the model to learn different
        spatial relationship patterns.
        
    Spatial Convolution Process:
        1. For each subset i, extract adjacency matrix A[i] of shape (V, V)
        2. Reshape input x from (N, C, T, V) to (N, C*T, V) for matrix multiplication
        3. Perform batch matrix multiplication: A[i] @ x → aggregates features from connected joints
        4. Apply learnable 1x1 convolution to transform aggregated features
        5. Sum outputs from all subsets to get final spatial features
        
    From STGCN paper:
        "The spatial graph convolution operation aggregates the feature information of 
        all neighboring nodes to update the node representation."
    
    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        A (torch.FloatTensor): Multi-layer adjacency matrix of shape (num_subset, V, V)
            containing normalized adjacency matrices for each spatial subset.
        num_subset (int): Number of spatial subsets (default: 3 for self/inward/outward).
    
    Input:
        x (torch.Tensor): Input feature tensor of shape (N, C, T, V) where:
            - N: batch size
            - C: number of input channels
            - T: number of temporal frames
            - V: number of joints/vertices
    
    Output:
        torch.Tensor: Output feature tensor of shape (N, out_channels, T, V) with
            spatially aggregated features from neighboring joints.
    
    Key Features:
        - Multi-subset spatial convolution for diverse spatial relationship modeling
        - Residual connection to prevent degradation (from STGCN++)
        - Learnable transformation via 1x1 convolutions after spatial aggregation
        - Supports arbitrary skeleton topologies through adjacency matrix A
    
    Example:
        ```python
        # Create adjacency matrix for skeleton graph
        graph = GraphAAGCN(edge_index, num_nodes)
        A = graph.A  # Shape: (3, V, V)
        
        # Create spatial unit
        gcn_unit = Basic_GCN_Unit(
            in_channels=64,
            out_channels=128,
            A=A,
            num_subset=3
        )
        
        # Input: batch_size=32, channels=64, frames=100, joints=25
        x = torch.randn(32, 64, 100, 25)
        output = gcn_unit(x)  # Shape: (32, 128, 100, 25)
        ```
    
    Note:
        - The 1x1 convolution kernel size is used because spatial aggregation is handled
          by matrix multiplication with the adjacency matrix
        - Each subset learns different spatial relationship patterns
        - Residual connection helps with gradient flow and model stability
        - Batch normalization and ReLU activation are applied after aggregation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.FloatTensor,
        num_subset: int = 3
    ):
        super(Basic_GCN_Unit, self).__init__()

        # The neighbors of a vertex are divided into num_subset subsets, 
        # 3 in the paper : self, inward and outward
        self.num_subset = num_subset 
        
        # A is the three layered adjacency matrix of the graph,
        self.A = A

        # one conv per subset
        self.conv_d = nn.ModuleList()

        for _ in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))
                 

        # skip connection in order to avoid the degradation problem (ResNet architecture)
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        self._init_conv_bn()

    def _conv_branch_init(self, conv, branches):
        weight = conv.weight
        n = weight.size(0)
        k1 = weight.size(1)
        k2 = weight.size(2)
        nn.init.normal_(weight, 0, math.sqrt(2.0 / (n * k1 * k2 * branches)))
        nn.init.constant_(conv.bias, 0)

    def _init_conv_bn(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        for i in range(self.num_subset):
            self._conv_branch_init(self.conv_d[i], self.num_subset)

    def _non_adaptive_forward(self, x, y):
        N, C, T, V = x.size()
        for i in range(self.num_subset):
            A1 = self.A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            # the result y is the sum of the 3 convolutions coming from the 3 subsets
            y = z + y if y is not None else z
        return y

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        y = self._non_adaptive_forward(x, y)
        y = self.bn(y)
        #skip connection technique
        y = y + self.down(x)
        y = self.relu(y)
        return y
