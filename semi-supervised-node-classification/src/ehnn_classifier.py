"""
Code adapted from: Equivariant Hypergraph Neural Networks, 2022,
 Jinwoo Kim and Saeyoon Oh and Sungjun Cho and Seunghoon Hong.
 Available from: https://github.com/jw9730/ehnn
 Article: https://arxiv.org/abs/2208.10428
 Accessed: 30 September 2023
"""

import ast

import torch.nn as nn

from ehnn.models import EHNNTransformer


class EHNNClassifier(nn.Module):
    def __init__(self, args, ehnn_cache):
        super().__init__()
        edge_orders = ehnn_cache["edge_orders"]  # [|E|,]
        overlaps = ehnn_cache["overlaps"]  # [|overlaps|,]
        max_edge_order = int(edge_orders.max().item())
        max_overlap = int(overlaps.max().item()) if overlaps is not None else 0
        hypernet_info = (max_edge_order, max_edge_order, max_overlap)

        self.model = EHNNTransformer(
            args.num_features,
            args.num_classes,
            args.ehnn_hidden_channel,
            args.ehnn_n_layers,
            args.ehnn_qk_channel,
            args.ehnn_hidden_channel,
            args.ehnn_n_heads,
            args.dropout,
            hypernet_info,
            args.ehnn_inner_channel,
            args.ehnn_pe_dim,
            args.ehnn_hyper_dim,
            args.ehnn_hyper_layers,
            args.ehnn_hyper_dropout,
            args.ehnn_input_dropout,
            ast.literal_eval(args.ehnn_mlp_classifier),
            args.Classifier_hidden,
            args.Classifier_num_layers,
            args.normalization,
            args.ehnn_att0_dropout,
            args.ehnn_att1_dropout,
        )

    def reset_parameters(self):
        self.model.reset_parameters()

    def forward(self, data, ehnn_cache, augmented_g=None):
        """forward method
        :param data:
        :param ehnn_cache:
        :return: [N, C] dense
        """
        x = data.x
        x, learner_weights = self.model(x, ehnn_cache, augmented_g)
        return x, learner_weights
