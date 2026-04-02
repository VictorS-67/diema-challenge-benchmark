import torch
import torch.nn as nn
from emo_mocap.models.stgcn.spatial_units import Basic_GCN_Unit
from emo_mocap.models.stgcn.tempo_units import Basic_TCN_Unit, TCN_Unit_plus

class STGCN_Unit(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network Unit for skeleton-based action recognition.
    
    This module combines spatial and temporal convolutions to capture both joint relationships
    within frames and temporal dependencies across frames. It represents the fundamental
    building block of ST-GCN models, processing motion capture sequences through sequential
    spatial and temporal convolution operations.
    
    Architecture:
        The unit follows a two-stage processing pipeline:
        1. **Spatial Convolution (GCN)**: Aggregates features from neighboring joints
           within each frame using the skeleton's graph structure
        2. **Temporal Convolution (TCN)**: Captures temporal dependencies across frames
           for each joint independently
        3. **Residual Connection**: Adds skip connection to prevent gradient degradation
    
    From ST-GCN paper:
        "The ST-GCN block applies graph convolution over the spatial configuration 
        and applies temporal convolution over the temporal axis, followed by a 
        residual connection."
    
    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        A (torch.FloatTensor): Multi-layer adjacency matrix of shape (num_subset, V, V)
            containing normalized adjacency matrices for spatial graph convolution.
        plusplus (bool): If True, uses advanced TCN_Unit_plus for temporal processing.
            If False, uses basic TCN_Unit. Default: False.
        stride (int): Temporal stride for downsampling along time dimension.
            Default: 1 (no temporal downsampling).
        residual (bool): If True, applies residual connection. If False, no skip connection.
            Default: True.
    
    Input:
        x (torch.Tensor): Input feature tensor of shape (N, C, T, V) where:
            - N: batch size
            - C: number of input channels
            - T: number of temporal frames
            - V: number of joints/vertices
    
    Output:
        torch.Tensor: Output feature tensor of shape (N, out_channels, T', V) where:
            - T' depends on stride (T' = T when stride=1)
    
    Processing Flow:
        1. **Spatial Processing**: x → GCN → spatially aggregated features
        2. **Temporal Processing**: spatially aggregated features → TCN → spatio-temporal features
        3. **Residual Addition**: spatio-temporal features + residual(x) → final output
        4. **Activation**: Apply ReLU activation
    
    Residual Connection Types:
        - **No residual**: residual(x) = 0
        - **Identity**: residual(x) = x (when in_channels == out_channels and stride == 1)
        - **Projection**: residual(x) = TCN_1x1(x) (when dimensions don't match)
    
    Example:
        ```python
        # Create adjacency matrix for skeleton graph
        graph = GraphAAGCN(edge_index, num_nodes)
        A = graph.A  # Shape: (3, V, V)
        
        # Basic ST-GCN unit
        st_unit = STGCN_Unit(
            in_channels=3,
            out_channels=64,
            A=A,
            plusplus=False,
            stride=1,
            residual=True
        )
        
        # Advanced ST-GCN++ unit
        st_unit_plus = STGCN_Unit(
            in_channels=64,
            out_channels=128,
            A=A,
            plusplus=True,
            stride=2,
            residual=True
        )
        
        # Input: batch_size=32, channels=3, frames=100, joints=25
        x = torch.randn(32, 3, 100, 25)
        output = st_unit(x)  # Shape: (32, 64, 100, 25)
        ```
    
    Note:
        - The spatial convolution must precede temporal convolution for proper
          spatio-temporal feature learning
        - plusplus=True enables multi-scale temporal processing for complex patterns
        - Residual connections are crucial for training deep ST-GCN networks
        - The adjacency matrix A defines the skeleton topology and must match
          the number of joints V in the input data
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.FloatTensor,
        plusplus: bool = False,
        stride: int = 1,
        residual: bool = True,
        unit_dropout: float = 0.3
    ):
        super(STGCN_Unit, self).__init__()

        # Initialize the GCN unit
        self.gcn1 = Basic_GCN_Unit(in_channels, out_channels, A)

        # Initialize the TCN unit
        if not plusplus:
            #if not plusplus, the basic TCN_Unit is used
            self.tcn1 = Basic_TCN_Unit(out_channels, out_channels, stride=stride)
        else:
            #if plusplus, the multi branch TCN_Unit_plus is used
            self.tcn1 = TCN_Unit_plus(out_channels, out_channels, dropout=unit_dropout, stride=stride)

        self.relu = nn.ReLU(inplace=True)

        # skip connection is added to regulate the degrading gradient problem
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = Basic_TCN_Unit(
                in_channels, out_channels, kernel_size=1, stride=stride
            )

    def forward(self, x):
        """
        Forward pass through the Spatio-Temporal Graph Convolutional Unit.
    
        The processing follows the ST-GCN architecture:
        spatial aggregation → temporal modeling → residual addition → activation.
        
        Args:
            x (torch.Tensor): Input feature tensor of shape (N, C, T, V) where:
                - N: batch size
                - C: number of input channels (must match in_channels)
                - T: number of temporal frames
                - V: number of joints/vertices
        
        Returns:
            torch.Tensor: Output feature tensor of shape (N, out_channels, T', V) where:
                - T' = T // stride (temporal dimension after stride)
                - All other dimensions preserve spatial relationships
        """
        y = self.gcn1(x)
        y = self.tcn1(y)
        y = y + self.residual(x)
        y = self.relu(y)
        return y