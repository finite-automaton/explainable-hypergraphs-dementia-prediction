"""
Code adapted from: Equivariant Hypergraph Neural Networks, 2022,
 Jinwoo Kim and Saeyoon Oh and Sungjun Cho and Seunghoon Hong.
 Available from: https://github.com/jw9730/ehnn
 Article: https://arxiv.org/abs/2208.10428
 Accessed: 30 September 2023
"""

from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, n_layers, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.f = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        assert n_layers > 0
        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(nn.Linear(dim_in, dim_out))
        else:
            for layer_idx in range(n_layers):
                self.layers.append(
                    nn.Linear(
                        dim_hidden if layer_idx > 0 else dim_in,
                        dim_hidden if layer_idx < n_layers - 1 else dim_out,
                    )
                )

        self.norms = nn.ModuleList()
        for layer_idx in range(n_layers - 1):
            self.norms.append(nn.LayerNorm(dim_hidden if layer_idx > 0 else dim_in))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(self.n_layers - 1):
            # x = self.norms[idx](x)
            x = self.layers[idx](x)
            x = self.f(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x

    def multi_forward(self, xs: List) -> List:
        for x in xs:
            assert len(x.size()) == 2
        return self.forward(torch.cat(xs)).split([x.size(0) for x in xs])