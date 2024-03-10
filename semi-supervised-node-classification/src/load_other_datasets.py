"""
Code adapted from: Equivariant Hypergraph Neural Networks, 2022,
 Jinwoo Kim and Saeyoon Oh and Sungjun Cho and Seunghoon Hong.
 Available from: https://github.com/jw9730/ehnn
 Article: https://arxiv.org/abs/2208.10428
 Accessed: 30 September 2023
"""

import os.path as osp

import numpy as np
import scipy.sparse as sp
import torch

from torch_geometric.data import Data
from torch_sparse import coalesce


def load_LE_dataset(path=None, dataset="nacc", train_percent=0.025):
    # 1. LOAD DATASET
    print("Loading {} dataset...".format(dataset))
    file_name = f"{dataset}.content"
    # path to nacc.content
    p2idx_features_labels = osp.join(path, file_name)
    # load into numpy format
    idx_features_labels = np.genfromtxt(p2idx_features_labels, dtype=np.dtype(str))

    # 2. EXTRACT FEATURES AND LABELS

    # creates a Compressed Sparse Row matrix from the features (assume last column is label)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # creates a tensor from the labels (classes)
    labels = torch.LongTensor(idx_features_labels[:, -1].astype(float))
    print("load features")
    # build graph
    # extract node indexes (node identifiers as an array)
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # map node indices to corresponding positions in array
    idx_map = {j: i for i, j in enumerate(idx)}
    # load edges relations (|V| -> |E|)
    file_name = f"{dataset}.edges"
    p2edges_unordered = osp.join(path, file_name)
    edges_unordered = np.genfromtxt(p2edges_unordered, dtype=np.int32)
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
    ).reshape(edges_unordered.shape)
    print("load edges")
    # Transpose, so we end up with a data format like:
    # [[node1, node1, node1, node2 ... Nn]
    #  [edge3, edge6, edge7, edge3 ... Ex]]
    edge_index = edges.T

    assert edge_index[0].max() == edge_index[1].min() - 1
    assert len(np.unique(edge_index)) == edge_index.max() + 1

    num_nodes = edge_index[0].max() + 1
    num_he = edge_index[1].max() - num_nodes + 1

    # stack edge index with its reverse so the graph is undirected
    edge_index = np.hstack((edge_index, edge_index[::-1, :]))
    data = Data(
        x=torch.FloatTensor(np.array(features[:num_nodes].todense())),
        edge_index=torch.LongTensor(edge_index),
        y=labels[:num_nodes],
    )

    total_num_node_id_he_id = len(np.unique(edge_index))
    data.edge_index, data.edge_attr = coalesce(
        data.edge_index, None, total_num_node_id_he_id, total_num_node_id_he_id
    )

    n_x = num_nodes
    data.n_x = n_x
    data.train_percent = train_percent
    data.num_hyperedges = num_he

    return data
