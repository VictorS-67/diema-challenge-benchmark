import torch
import torch.nn as nn
from emo_mocap.models.weights_init import conv_init, bn_init

class Basic_TCN_Unit(nn.Module):
    """
    Basic Temporal Convolutional Network Unit for skeleton-based action recognition.
    https://github.com/yysijie/st-gcn/

    This module performs temporal convolution on motion capture sequences to aggregate
    information from neighboring frames. Unlike spatial graph convolution that operates
    on joint relationships within a frame, this unit captures temporal dependencies
    by convolving across the time dimension.
    
    Architecture:
        The unit applies a 1D temporal convolution (implemented as 2D with kernel_size=(Kt, 1))
        across the time dimension while preserving spatial relationships. Each joint's
        features are aggregated with its temporal neighbors (past and future frames).
        
    From STGCN paper:
        "For the temporal dimension, since the number of neighbors for each vertex is 
        fixed as 2 (corresponding joints in the previous and following frames), it is 
        straightforward to perform the graph convolution similar to the classical 
        convolution operation."
    
    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        kernel_size (int): Temporal kernel size (number of frames to aggregate).
            Default: 9 (captures 4 past + current + 4 future frames).
        stride (int): Temporal stride for downsampling along time dimension.
            Default: 1 (no temporal downsampling).
        dilation (int): Temporal dilation for expanding receptive field.
            Default: 1 (no dilation).
    
    Input:
        x (torch.Tensor): Input feature tensor of shape (N, C, T, V) where:
            - N: batch size
            - C: number of input channels
            - T: number of temporal frames
            - V: number of joints/vertices
    
    Output:
        torch.Tensor: Output feature tensor of shape (N, out_channels, T', V) where:
            - T' depends on stride (T' = T when stride=1)
    
    Note:
        - Padding is automatically calculated to maintain temporal dimension when stride=1
        - Convolution operates only along time dimension (kernel_size=(Kt, 1))
        - Batch normalization and ReLU activation are applied after convolution
        - Weight initialization is performed using conv_init()

    Example:
        ```python
        # Create temporal unit for 64->128 channel transformation
        tcn_unit = Basic_TCN_Unit(
            in_channels=64,
            out_channels=128,
            kernel_size=9,
            stride=1
        )
        
        # Input: batch_size=32, channels=64, frames=100, joints=25
        x = torch.randn(32, 64, 100, 25)
        output = tcn_unit(x)  # Shape: (32, 128, 100, 25)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dilation: int = 1
        ):
        super(Basic_TCN_Unit, self).__init__()
        # when dilation = 1 : pad = int((kernel_size - 1) / 2) 
        pad = int((kernel_size + (kernel_size - 1) * (dilation - 1) - 1) / 2)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1), 
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1)
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv) #initialize the weights
        bn_init(self.bn, 1) #initialize the weights

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x
    




class TCN_Unit_plus(nn.Module):
    """
    Multi-branch Temporal Convolutional Network Unit from STGCN++ (PySKL).
    
    This advanced temporal unit implements an Inception-style multi-branch architecture
    that captures temporal dependencies at multiple scales simultaneously. Instead of
    using a single temporal convolution, it splits the computation across six parallel
    branches, each capturing different temporal patterns and receptive fields.
    
    Architecture:
        The unit consists of six parallel branches:
        1. One 1x1 convolution branch (direct feature transformation)
        2. One max-pooling branch (temporal downsampling with feature extraction)
        3. Four temporal convolution branches with kernel size 3 and dilations 1-4
           (multi-scale temporal feature extraction)
        
        Each branch processes a portion of the input channels, and their outputs are
        concatenated and refined through a final 1x1 convolution.
    
    From PySKL paper:
        "The adopted multi-branch TCN consists of six branches: a '1x1' Conv branch, 
        a Max-Pooling branch, and four temporal 1D Conv branches with kernel size 3 
        and dilations from 1 to 4. It first transforms features with '1x1' Conv and 
        divides them into six groups with equal channel width. Then, each feature 
        group is processed with a single branch. The six outputs are concatenated 
        together and processed by another '1x1' Conv to form the output of the 
        multi-branch TCN."
    
    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        dropout (float): Dropout probability for regularization. Default: 0.0.
        stride (int): Temporal stride for downsampling along time dimension.
            Default: 1 (no temporal downsampling).
    
    Input:
        x (torch.Tensor): Input feature tensor of shape (N, C, T, V) where:
            - N: batch size
            - C: number of input channels (must equal in_channels)
            - T: number of temporal frames
            - V: number of joints/vertices
    
    Output:
        torch.Tensor: Output feature tensor of shape (N, out_channels, T', V) where:
            - T' depends on stride (T' = T when stride=1)
    
    Channel Distribution:
        - Branch 0: rem_mid_channels = out_channels - mid_channels * 5
        - Branches 1-5: mid_channels = out_channels // 6
        - Total output channels = rem_mid_channels + 5 * mid_channels = out_channels
    
    Branch Configuration:
        - Branch 0: 3x1 conv with dilation=1 (local temporal patterns)
        - Branch 1: 3x1 conv with dilation=2 (medium-range temporal patterns)
        - Branch 2: 3x1 conv with dilation=3 (longer-range temporal patterns)
        - Branch 3: 3x1 conv with dilation=4 (long-range temporal patterns)
        - Branch 4: Max pooling with kernel=3 (temporal downsampling)
        - Branch 5: 1x1 conv (channel transformation only)
    
    Advantages:
        - Captures multi-scale temporal dependencies simultaneously
        - More computationally efficient than single large kernel
        - Better gradient flow through multiple parallel paths
        - Increased model capacity with controlled parameter growth

    Example:
        ```python
        # Create multi-branch temporal unit
        tcn_plus = TCN_Unit_plus(
            in_channels=64,
            out_channels=128,
            dropout=0.1,
            stride=1
        )
        
        # Input: batch_size=32, channels=64, frames=100, joints=25
        x = torch.randn(32, 64, 100, 25)
        output = tcn_plus(x)  # Shape: (32, 128, 100, 25)
        ```

    Note:
        - Each branch processes the full input but outputs different channel counts
        - Batch normalization and ReLU are applied within each branch
        - Final transformation includes additional batch norm and dropout
        - More suitable for complex temporal patterns than Basic_TCN_Unit
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout: float = 0.3,
                 stride: int = 1
                 ) -> None:

        super().__init__()
        # Multiple branches of temporal convolution
        # multi scale config:
        ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1']
        num_branches = len(ms_cfg)
        self.relu = nn.ReLU()
        
        mid_channels = out_channels // num_branches
        rem_mid_channels = out_channels - mid_channels * (num_branches - 1)

        branches = []
        for i, cfg in enumerate(ms_cfg):
            #branch_c is the number of out_channels for each branch
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                # -- 1x1 conv
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                # -- max pool
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1),
                        nn.BatchNorm2d(branch_c),
                        self.relu,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            # -- normal conv with dilation
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1),
                nn.BatchNorm2d(branch_c),
                self.relu,
                Basic_TCN_Unit(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1]))
            branches.append(branch)
            
        self.branches = nn.ModuleList(branches)

        self.transform = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            self.relu,
            nn.Conv2d(out_channels, out_channels, kernel_size=1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

        # Add weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for layers not covered by Basic_TCN_Unit"""
        for branch in self.branches:
            if isinstance(branch, nn.Conv2d):  # 1x1 conv branch
                conv_init(branch)
            elif isinstance(branch, nn.Sequential):
                for layer in branch:
                    if isinstance(layer, nn.Conv2d):
                        conv_init(layer)
                    elif isinstance(layer, nn.BatchNorm2d):
                        bn_init(layer, 1)
        
        # Initialize transform layers
        for layer in self.transform:
            if isinstance(layer, nn.Conv2d):
                conv_init(layer)
            elif isinstance(layer, nn.BatchNorm2d):
                bn_init(layer, 1)
        
        # Initialize final batch norm
        bn_init(self.bn, 1)

    def inner_forward(self, x):
        # apply the conv branch by branch, and concatenate the results
        N, C, T, V = x.shape

        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        feat = torch.cat(branch_outs, dim=1)
        feat = self.transform(feat)
        return feat

    def forward(self, x):
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)
    
