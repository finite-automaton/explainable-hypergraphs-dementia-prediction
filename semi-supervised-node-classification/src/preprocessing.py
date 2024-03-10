"""
Code adapted from: Equivariant Hypergraph Neural Networks, 2022,
 Jinwoo Kim and Saeyoon Oh and Sungjun Cho and Seunghoon Hong.
 Available from: https://github.com/jw9730/ehnn
 Article: https://arxiv.org/abs/2208.10428
 Accessed: 30 September 2023
"""

import numpy as np
import torch


def ConstructH(data):
    """
    Construct incidence matrix H of size (num_nodes,num_hyperedges) from edge_index = [V;E]
    """
    edge_index = np.array(data.edge_index)
    num_nodes = data.x.shape[0]
    num_hyperedges = np.max(edge_index[1]) - np.min(edge_index[1]) + 1
    H = np.zeros((num_nodes, num_hyperedges))
    cur_idx = 0
    for he in np.unique(edge_index[1]):
        nodes_in_he = edge_index[0][edge_index[1] == he]
        H[nodes_in_he, cur_idx] = 1.0
        cur_idx += 1

    data.edge_index = H
    return data


def ConstructHSparse(data):
    """
    Construct incidence matrix H of size (num_nodes,num_hyperedges) from edge_index = [V;E]
    """
    edge_index = np.array(data.edge_index)
    num_nodes = data.x.shape[0]
    num_hyperedges = np.max(edge_index[1]) - np.min(edge_index[1]) + 1
    edge_index[1] = edge_index[1] - num_nodes
    assert np.min(edge_index[1]) == 0 and np.max(edge_index[1]) == num_hyperedges - 1
    data.edge_index = edge_index
    return data


def ExtractV2E(data):
    # Assume edge_index = [V|E;E|V]
    edge_index = data.edge_index
    # First, ensure the sorting is correct (increasing along edge_index[0])
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)

    num_nodes = data.n_x[0] if isinstance(data.n_x, list) else data.n_x
    num_hyperedges = (
        data.num_hyperedges[0]
        if isinstance(data.num_hyperedges, list)
        else data.num_hyperedges
    )

    if not ((num_nodes + num_hyperedges - 1) == data.edge_index[0].max().item()):
        print("num_hyperedges does not match! 1")
        return
    cidx = torch.where(edge_index[0] == num_nodes)[0].min()  # cidx: [V...|cidx E...]
    data.edge_index = edge_index[:, :cidx].type(torch.LongTensor)
    return data


def rand_train_test_idx(label, train_prop=0.6, valid_prop=0.2):
    indices = []
    for i in range(label.max() + 1):
        index = (label == i).nonzero(as_tuple=True)[0]
        indices.append(index[torch.randperm(len(index))])

    train_idx_list, valid_idx_list, test_idx_list = [], [], []

    for idx in indices:
        n = len(idx)
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        train_idx_list.append(idx[:train_num])
        valid_idx_list.append(idx[train_num:train_num + valid_num])
        test_idx_list.append(idx[train_num + valid_num:])

    train_idx = torch.cat(train_idx_list)
    valid_idx = torch.cat(valid_idx_list)
    test_idx = torch.cat(test_idx_list)
    return {"train": train_idx, "valid": valid_idx, "test": test_idx}
