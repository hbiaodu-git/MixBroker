# coding=utf-8
import pandas as pd
import time
import os
import numpy as np


def ground_truth_generate():
    start = time.perf_counter()
    df_1 = pd.read_csv('./Dataset/Ens/Transfer_OnlyOne.csv', usecols=[4, 6])
    df_2 = pd.read_csv('./Dataset/Ens/NewOwner_OnlyOne.csv', usecols=[4, 6])
    df_1 = df_1.append(df_2).drop_duplicates(subset=['From', 'New']).reset_index().drop(columns=['index'])
    df_1.columns = ['From', 'To']
    df_3 = pd.read_csv('./Dataset/Graph/ETH.csv')
    df_3_deposit = df_3[df_3['Method'] == 'Deposit']
    df_3_withdraw = df_3[df_3['Method'] == 'Withdraw']
    i = 0
    arr_1 = [['node1', 'node2']]
    arr_2 = [['node1', 'node2']]
    while i < len(df_1.index):
        ac_1 = df_1['From'][i]
        ac_2 = df_1['To'][i]
        tx_1 = df_3_withdraw[df_3_withdraw['From'] == ac_1]
        tx_2 = df_3_deposit[df_3_deposit['From'] == ac_2]
        if not tx_1.empty and not tx_2.empty:
            temp_arr = [ac_2, ac_1]
            arr_1.insert(1, temp_arr)
        tx_3 = df_3_withdraw[df_3_withdraw['From'] == ac_2]
        tx_4 = df_3_deposit[df_3_deposit['From'] == ac_1]
        if not tx_3.empty and not tx_4.empty:
            temp_arr = [ac_1, ac_2]
            arr_2.insert(1, temp_arr)
        i += 1
        if i % 100 == 0:
            print('%s data have been processed!' % i)
    arr_1 = np.delete(arr_1, 0, axis=0)
    arr_2 = np.delete(arr_2, 0, axis=0)
    df_new_1 = pd.DataFrame(arr_1, columns=['node1', 'node2'])
    df_new_2 = pd.DataFrame(arr_2, columns=['node1', 'node2'])

    df_new_1 = df_new_1.append(df_new_2).drop_duplicates().reset_index().drop(columns=['index'])
    print(df_new_1)
    df_new.to_csv('./Dataset/Graph/ens_edge.csv')
    end = time.perf_counter()
    print('Total Time: %s' % (end - start))


def find_nodeid():
    start = time.perf_counter()
    df_node = pd.read_csv('./Dataset/Graph/node_feature_normalized.csv')
    df_edge = pd.read_csv('./Dataset/Graph/ens_edge.csv', usecols=[1, 2])
    df_edge = df_edge.drop_duplicates().reset_index().drop(columns=['index'])
    print(df_edge)
    arr = [['node1', 'node2']]
    i = 0
    while i < len(df_edge):
        sender = df_edge['Sender'][i]
        receiver = df_edge['Receiver'][i]
        node1 = df_node[df_node['node'] == sender].iloc[0, 0]
        node2 = df_node[df_node['node'] == receiver].iloc[0, 0]
        temp_arr = [node1, node2]
        arr.insert(1, temp_arr)
        i += 1
    arr = np.delete(arr, 0, axis=0)
    df_new = pd.DataFrame(arr, columns=['nodeid1', 'nodeid2'])
    print(df_new)
    df_new.to_csv('./Dataset/Graph/train_pos_edge_10fold.csv')
    end = time.perf_counter()
    print('Total Time: %s' % (end - start))


if __name__ == '__main__':
    # ground_truth_generate()
    find_nodeid()