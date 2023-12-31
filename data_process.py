# coding=utf-8
import pandas as pd
import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data


def load_10_fold_data(t_edge):
    t_data = pd.read_csv('./Dataset/Graph/node_feature_normalized.csv')
    t_data = t_data.drop(columns=['node', 'value_d', 'value_w', 'avg_value_d', 'avg_value_w'])

    names = t_data.columns.tolist()
    names = np.delete(names, 0)

    texId2index = {}
    for index, row in t_data.iterrows():
        texId2index[int(row.iloc[0])] = index

    x = t_data.iloc[:, 1:]
    x = x.reset_index(drop=True)
    x = x.to_numpy().astype(np.float32)

    x[x == np.inf] = 1.
    x[np.isnan(x)] = 0.

    edges = []
    labels = []
    for _, row in t_edge.iterrows():
        id_1, id_2 = int(row.iloc[0]), int(row.iloc[1])
        label = int(row.iloc[2])
        if id_1 not in texId2index or id_2 not in texId2index:
            continue
        edges.append((texId2index[id_1], texId2index[id_2]))
        labels.append(label)
    x = torch.tensor(x, dtype=torch.float32)
    d_edges = np.array(edges)
    edges = torch.tensor(d_edges.T, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32)

    data = Data(x=x, edge_index=edges, edge_label=labels, edge_label_index=edges)
    return data, names


if __name__ == '__main__':
    load_data()