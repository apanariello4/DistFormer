import torch.nn as nn
import torch.nn.functional as F
import math

import torch

from torch.nn.parameter import Parameter

from torch_geometric.nn import GATv2Conv


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias

        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(input_dim, input_dim)
        # self.gc2 = GraphConvolution(hidden_dim, input_dim)

    def forward(self, x, adj):
        x = F.elu(self.gc1(x, adj))
        # x = self.gc2(x, adj)

        return x


class SinkhornKnopp(nn.Module):
    def forward(self, P: torch.Tensor, num_iter: int = 5):
        N = P.shape[0]
        for i in range(num_iter):
            P = P / P.sum(1).view((N, 1))
            P = P / P.sum(0).view((1, N))
        return P


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.graph_norm = SinkhornKnopp()

    def forward(self, input, adj):
        # Eq. 8
        h = torch.mm(input, self.W)  # matrix multiplication of the matrices

        # Eq. 11
        N = h.size()[0]
        e = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(
            N, -1, 2 * self.out_features
        )
        e = torch.matmul(e, self.a).squeeze(2)
        e = self.leakyrelu(e)
        e = torch.exp(e)

        # Eq. 10
        alfa = self.graph_norm(e * adj)

        # Dropout
        # alfa = nn.Dropout(p=0.2)(alfa)

        # Eq. 7
        h_prime = torch.matmul(alfa, h)

        return h_prime, alfa


class GAT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ):
        super(GAT, self).__init__()

        # self.gc1 = GraphAttentionLayer(input_dim, input_dim, alpha=0.2, concat=True)
        self.gc1 = GATv2Conv(input_dim, input_dim, heads=4, dropout=0.2)
        # self.gc2 = GraphConvolution(hidden_dim, input_dim)

    def forward(self, x, adj):
        x = self.gc1(x, adj)

        # x = self.gc2(x, adj)

        return x


class StackedGATv2(nn.Module):
    def __init__(
        self,
        node_feats,
        n_layers,
        n_heads,
        n_hidden,
        dropout,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_hidden = n_hidden

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.node_encoder = nn.Linear(node_feats, n_hidden)

        for i in range(n_layers):
            in_hidden = n_hidden
            out_hidden = n_hidden

            layer = GATv2Conv(
                in_hidden,
                out_hidden // n_heads,
                heads=n_heads,
            )
            self.convs.append(layer)
            self.norms.append(nn.LayerNorm(out_hidden))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.node_encoder(x)
        h = F.relu(h, inplace=True)

        for i in range(self.n_layers):
            h = self.convs[i](h, edge_index)
            h = self.norms[i](h)
            h = F.elu(h, inplace=True)
            h = self.dropout(h)

        return h


if __name__ == "__main__":
    # Test the model
    model = GCN(input_dim=512, hidden_dim=256, output_dim=128, residual=True)
    x = torch.randn(12, 1024)
    adj = torch.randn(12, 12)
    output = model(x, adj)
    print(output)
