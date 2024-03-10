"""
Code adapted from: Equivariant Hypergraph Neural Networks, 2022,
 Jinwoo Kim and Saeyoon Oh and Sungjun Cho and Seunghoon Hong.
 Available from: https://github.com/jw9730/ehnn
 Article: https://arxiv.org/abs/2208.10428
 Accessed: 30 September 2023
"""

import math

import torch
import torch.nn as nn

from .hypernetwork import PositionalMLP
class BiasE(nn.Module):
    def __init__(self, dim_out,
                 max_l, pe_dim, hyper_dim, hyper_layers, hyper_dropout):
        super().__init__()
        self.dim_out = dim_out
        self.max_l = max_l
        self.pe_dim = pe_dim
        self.hyper_dim = hyper_dim
        self.hyper_layers = hyper_layers
        self.hyper_dropout = hyper_dropout
        self.b = PositionalMLP(dim_out, max_l + 1, pe_dim, hyper_dim, hyper_layers, hyper_dropout)

    def reset_parameters(self):
        self.b.reset_parameters()

    def forward(self, x, edge_orders):
        x_v, x_e = x
        if self.max_l and self.hyper_dropout == 0:
            # do not use this when hypernetwork is stochastic
            indices = torch.arange(self.b.max_pos, device=x_v.device)
            b = self.b(indices)[edge_orders]
        else:
            b = self.b(edge_orders)
        b_1 = self.b(torch.ones(1, dtype=torch.long, device=x_v.device)).view(1, self.dim_out)  # [D']
        return x_v + b_1, x_e + b


class BiasV(nn.Module):
    def __init__(self, dim_out):
        super().__init__()
        self.dim_out = dim_out
        self.b = nn.Parameter(torch.Tensor(1, dim_out))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dim_out)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return x + self.b