import torch
import torch.nn.functional as F

class GraphAAGCN:
    r"""
    Defining the Graph for the Two-Stream Adaptive Graph Convolutional Network.
    It's composed of the normalized inward-links, outward-links and
    self-links between the nodes as originally defined in the
    `authors repo  <https://github.com/lshiwjx/2s-AGCN/blob/master/graph/tools.py>`
    resulting in the shape of (3, num_nodes, num_nodes).
    Args:
        edge_index (Tensor array): Edge indices
        num_nodes (int): Number of nodes
    Return types:
            * **A** (PyTorch Float Tensor) - Three layer normalized adjacency matrix 
            following the pattern [self-links, inward-links, outward-links]
    """

    def __init__(self, edge_index: list, num_nodes: int):
        self.num_nodes = num_nodes
        self.edge_index = torch.tensor(edge_index)
        self.A = self.get_spatial_graph(self.num_nodes)

    def get_spatial_graph(self, num_nodes):
        self_mat = torch.eye(num_nodes) #id matrix
        inward_mat = to_dense_adj(self.edge_index, num_nodes) #adjacency matrix
        inward_mat_norm = F.normalize(inward_mat, dim=0, p=1) #normalize the columns (sum = 1)
        outward_mat = inward_mat.transpose(0, 1) 
        outward_mat_norm = F.normalize(outward_mat, dim=0, p=1) #normalize the columns (sum = 1)
        adj_mat = torch.stack((self_mat, inward_mat_norm, outward_mat_norm))
        return adj_mat
    
def to_dense_adj(edge_index, num_nodes):
    #assume edge_index is a tensor already
    effective_num_nodes = torch.tensor([int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0])
    assert effective_num_nodes.item() <= num_nodes
    
    adj = torch.zeros((num_nodes, num_nodes))

    for (i, j) in edge_index:
        adj[i, j] = 1

    return adj