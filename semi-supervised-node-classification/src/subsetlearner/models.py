"""
Code adapted from: Counterfactual and Factual Reasoning over Hypergraphs for Interpretable Clinical Predictions on EHR, 2022,
 Xu, Ran and Yu, Yue and Zhang, Chao and Ali, Mohammed K and Ho, Joyce C and Yang, Carl
 Available from: https://github.com/ritaranx/CACHE
 Article: https://proceedings.mlr.press/v193/xu22a.html
 Accessed: 4 October 2023
"""

import torch

import torch.nn as nn
from subsetlearner.layers import *


class ViewLearner(torch.nn.Module):
    def __init__(self, input_dim, viewer_hidden_dim=64):
        super(ViewLearner, self).__init__()

        self.input_dim = input_dim

        self.mlp_edge_model = nn.Sequential(
            nn.LayerNorm(self.input_dim * 2),
            nn.Linear(self.input_dim * 2, viewer_hidden_dim),
            nn.ReLU(),
            nn.Linear(viewer_hidden_dim, 1)
        )
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, model, data, ehnn_cache):
        model.eval()
        with torch.no_grad():
            _, learner_features = model(data.clone(), ehnn_cache)
        node_feat, edge_feat = learner_features
        edge_index = data.edge_index.copy()
        node, edge = edge_index[0], edge_index[1]
        emb_node = node_feat[node]
        emb_edge = edge_feat[edge]
        combined = torch.cat([emb_edge, emb_node], 1)

        weights_e_v = self.mlp_edge_model(combined)

        return weights_e_v
