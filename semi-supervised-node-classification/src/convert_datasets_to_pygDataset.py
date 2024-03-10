"""
Code adapted from: Equivariant Hypergraph Neural Networks, 2022,
 Jinwoo Kim and Saeyoon Oh and Sungjun Cho and Seunghoon Hong.
 Available from: https://github.com/jw9730/ehnn
 Article: https://arxiv.org/abs/2208.10428
 Accessed: 30 September 2023
"""

import os
import os.path as osp
import pickle

import torch
from torch_geometric.data import InMemoryDataset

from load_other_datasets import load_LE_dataset


def save_data_to_pickle(data, p2root="../data/", file_name=None):
    """
    if file name not specified, use time stamp.
    """
    tmp_data_name = file_name
    path_name = osp.join(p2root, tmp_data_name)
    if not osp.isdir(p2root):
        os.makedirs(p2root)
    with open(path_name, "bw") as f:
        pickle.dump(data, f)
    return path_name


class dataset_Hypergraph(InMemoryDataset):
    def __init__(
        self,
        folder,
        root="../data/pyg_data/hypergraph_dataset_updated/",
        train_percent=0.6,
    ):

        self.name = folder
        self.p2raw = f"../data/{folder}/"

        self._train_percent = train_percent

        if not osp.isdir(self.p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{self.p2raw}" does not exist!'
            )

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root
        self.myraw_dir = osp.join(root, self.name, "raw")
        self.myprocessed_dir = osp.join(root, self.name, "processed_data")

        super(dataset_Hypergraph, self).__init__(osp.join(root, self.name))

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = self.data.train_percent

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        file_names = ["data.pt"]
        return file_names

    @property
    def num_features(self):
        return self.data.num_node_features

    def download(self):
        for name in self.raw_file_names:
            p2f = osp.join(self.myraw_dir, name)
            if not osp.isfile(p2f):
                # file not exist, so we create it and save it there.
                tmp_data = load_LE_dataset(
                    path=self.p2raw,
                    dataset="nacc",
                    train_percent=self._train_percent,
                )
                print(f"num_node: {tmp_data.n_x}")
                print(f"num_edge: {tmp_data.num_hyperedges}")

                _ = save_data_to_pickle(
                    tmp_data, p2root=self.myraw_dir, file_name=self.raw_file_names[0]
                )
            else:
                pass

    def process(self):
        p2f = osp.join(self.myraw_dir, self.raw_file_names[0])
        with open(p2f, "rb") as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return "{}()".format(self.name)
