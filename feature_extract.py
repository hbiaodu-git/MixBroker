# coding=utf-8
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import preprocessing


def concat_all():
    start = time.perf_counter()
    df1 = pd.read_csv('./Dataset/Concat/0.1ETH-concat.csv')
    df2 = pd.read_csv('./Dataset/External/0.1ETH-external-gas.csv', usecols=[1, 10])
    file_list = [1, 10, 100]
    for val in file_list:
        df3 = pd.read_csv('./Dataset/Concat/{}ETH-concat.csv'.format(val))
        df1 = df1.append(df3)
        df4 = pd.read_csv('./Dataset/External/{}ETH-external-gas.csv'.format(val), usecols=[1, 10])
        df2 = df2.append(df4)
    df1 = pd.merge(df1, df2, on='Txhash', how='outer')
    df1 = df1.sort_values(by=['From', 'To']).reset_index().drop(columns=['index', 'Unnamed: 0'])
    df1.to_csv('./Dataset/Graph/ETH.csv')
    # print(df1)
    end = time.perf_counter()
    print('Total Time: %s' % (end - start))


def unique_address():
    start = time.perf_counter()
    df1 = pd.read_csv('./Dataset/Graph/ETH.csv')
    df1_deposit = df1[df1['Method'] == 'Deposit']
    df1_withdraw = df1[df1['Method'] == 'Withdraw']
    print(len(set(df1_deposit['From'])))
    print(len(set(df1_withdraw['From'])))
    print(len(set(df1['From'])))

    end = time.perf_counter()
    print('Total Time: %s' % (end - start))


def feature_extract_value():
    start = time.perf_counter()
    df1 = pd.read_csv('./Dataset/Graph/ETH.csv')
    i = 0
    arr = [['node', '0.1_num', '1_num', '10_num', '100_num', 'num_all', '0.1_num_d', '0.1_num_w', '1_num_d', '1_num_w',
            '10_num_d', '10_num_w', '100_num_d', '100_num_w', 'num_d_all', 'num_w_all', 'value_d', 'value_w',
            'avg_value_d', 'avg_value_w']]

    while i < len(df1.index):
        j = i
        while j < len(df1.index) and df1['From'][j] == df1['From'][i]:
            j += 1

        num1, num2, num3, num4 = 0, 0, 0, 0
        num1_d, num1_w, num2_d, num2_w, num3_d, num3_w, num4_d, num4_w = 0, 0, 0, 0, 0, 0, 0, 0
        for k in range(i, j):
            if df1['To'][k] == '0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc':
                num1 += 1
                if df1['Method'][k] == 'Deposit':
                    num1_d += 1
                else:
                    num1_w += 1
            elif df1['To'][k] == '0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936':
                num2 += 1
                if df1['Method'][k] == 'Deposit':
                    num2_d += 1
                else:
                    num2_w += 1
            elif df1['To'][k] == '0x910cbd523d972eb0a6f4cae4618ad62622b39dbf':
                num3 += 1
                if df1['Method'][k] == 'Deposit':
                    num3_d += 1
                else:
                    num3_w += 1
            else:
                num4 += 1
                if df1['Method'][k] == 'Deposit':
                    num4_d += 1
                else:
                    num4_w += 1

        num_all = num1 + num2 + num3 + num4
        num_d_all = num1_d + num2_d + num3_d + num4_d
        num_w_all = num1_w + num2_w + num3_w + num4_w
        value_d = 0.1 * num1_d + 1 * num2_d + 10 * num3_d + 100 * num4_d
        value_w = 0.1 * num1_w + 1 * num2_w + 10 * num3_w + 100 * num4_w
        avg_value_d, avg_value_w = 0, 0
        if num_d_all != 0:
            avg_value_d = value_d / num_d_all
        if num_w_all != 0:
            avg_value_w = value_w / num_w_all
        temp_arr = [df1['From'][i], num1, num2, num3, num4, num_all, num1_d, num1_w, num2_d, num2_w, num3_d, num3_w,
                    num4_d, num4_w, num_d_all, num_w_all, value_d, value_w, avg_value_d, avg_value_w]
        arr.insert(1, temp_arr)
        i = j

        if i % 10000 == 0:
            print('%s data have been processed!' % i)

    arr = np.delete(arr, 0, axis=0)
    df_new = pd.DataFrame(arr, columns=['node', '0.1_num', '1_num', '10_num', '100_num', 'num_all', '0.1_num_d',
                                        '0.1_num_w', '1_num_d', '1_num_w', '10_num_d', '10_num_w', '100_num_d',
                                        '100_num_w', 'num_d_all', 'num_w_all', 'value_d', 'value_w', 'avg_value_d',
                                        'avg_value_w'])
    df_new = df_new.sort_values(by=['node']).reset_index().drop(columns=['index'])
    print(df_new)
    df_new.to_csv('./Dataset/Graph/node_feature.csv')
    end = time.perf_counter()
    print('Total Time: %s' % (end - start))


def feature_extract_time1():
    start = time.perf_counter()
    df1 = pd.read_csv('./Dataset/Graph/ETH.csv')
    df1 = df1.sort_values(by=['From', 'UnixTimestamp']).reset_index().drop(columns=['index', 'Unnamed: 0'])
    df2 = pd.read_csv('./Dataset/Graph/node_feature_copy.csv')
    df2 = df2.drop(columns=['Unnamed: 0', 'early_time', 'late_time', 'total_time_gap', 'min_time_gap', 'max_time_gap',
                            'avg_time_gap', 'early_time_d', 'late_time_d', 'total_time_gap_d', 'min_time_gap_d',
                            'max_time_gap_d', 'avg_time_gap_d', 'early_time_w', 'late_time_w', 'total_time_gap_w',
                            'min_time_gap_w', 'max_time_gap_w', 'avg_time_gap_w'])
    i = 0
    arr = [['node', 'early_time', 'late_time', 'total_time_gap', 'min_time_gap', 'max_time_gap', 'avg_time_gap']]
    while i < len(df1.index):
        j = i
        while j < len(df1.index) and df1['From'][j] == df1['From'][i]:
            j += 1
        total_time_arr = []
        if i == j - 1:
            total_time_arr = [0]
        else:
            for k in range(i, j - 1):
                time_gap = df1['UnixTimestamp'][k + 1] - df1['UnixTimestamp'][k]
                total_time_arr.append(time_gap)
        total_time_arr.sort()

        early_time = df1['UnixTimestamp'][i]
        late_time = df1['UnixTimestamp'][j - 1]
        total_time_gap = late_time - early_time
        min_time_gap = total_time_arr[0]
        max_time_gap = total_time_arr[-1]
        avg_time_gap = sum(total_time_arr) / len(total_time_arr)
        tem_arr = [df1['From'][i], early_time, late_time, total_time_gap, min_time_gap, max_time_gap, avg_time_gap]
        arr.insert(1, tem_arr)
        i = j
        if i % 10000 == 0:
            print('%s data have been processed!' % i)
    arr = np.delete(arr, 0, axis=0)
    df_new = pd.DataFrame(arr, columns=['node', 'early_time', 'late_time', 'total_time_gap', 'min_time_gap', 'max_time_gap', 'avg_time_gap'])
    print(df_new)
    df2 = pd.merge(df2, df_new, on='node', how='outer')
    df2.to_csv('./Dataset/Graph/node_feature_copy.csv')
    end = time.perf_counter()
    print('Total Time: %s' % (end - start))


def feature_extract_time2():
    start = time.perf_counter()
    df1 = pd.read_csv('./Dataset/Graph/ETH.csv')
    df1_deposit = df1[df1['Method'] == 'Deposit']
    df1_deposit = df1_deposit.sort_values(by=['From', 'UnixTimestamp']).reset_index().drop(columns=['index', 'Unnamed: 0'])
    df1_withdraw = df1[df1['Method'] == 'Withdraw']
    df1_withdraw = df1_withdraw.sort_values(by=['From', 'UnixTimestamp']).reset_index().drop(columns=['index', 'Unnamed: 0'])
    df2 = pd.read_csv('./Dataset/Graph/node_feature_copy.csv')
    df2 = df2.drop(columns=['Unnamed: 0'])
    i = 0
    arr_deposit = [['node', 'early_time_d', 'late_time_d', 'total_time_gap_d', 'min_time_gap_d', 'max_time_gap_d', 'avg_time_gap_d']]
    arr_withdraw = [['node', 'early_time_w', 'late_time_w', 'total_time_gap_w', 'min_time_gap_w', 'max_time_gap_w', 'avg_time_gap_w']]
    while i < len(df1_withdraw.index):
        j = i
        while j < len(df1_withdraw.index) and df1_withdraw['From'][j] == df1_withdraw['From'][i]:
            j += 1
        total_time_arr = []
        if i == j - 1:
            total_time_arr = [0]
        else:
            for k in range(i, j - 1):
                time_gap = df1_withdraw['UnixTimestamp'][k + 1] - df1_withdraw['UnixTimestamp'][k]
                total_time_arr.append(time_gap)
        total_time_arr.sort()

        early_time = df1_withdraw['UnixTimestamp'][i]
        late_time = df1_withdraw['UnixTimestamp'][j - 1]
        total_time_gap = late_time - early_time
        min_time_gap = total_time_arr[0]
        max_time_gap = total_time_arr[-1]
        avg_time_gap = sum(total_time_arr) / len(total_time_arr)
        tem_arr = [df1_withdraw['From'][i], early_time, late_time, total_time_gap, min_time_gap, max_time_gap, avg_time_gap]
        arr_withdraw.insert(1, tem_arr)
        i = j
        if i % 10000 == 0:
            print('%s data have been processed!' % i)
    arr_withdraw = np.delete(arr_withdraw, 0, axis=0)
    df_new = pd.DataFrame(arr_withdraw, columns=['node', 'early_time_w', 'late_time_w', 'total_time_gap_w', 'min_time_gap_w', 'max_time_gap_w', 'avg_time_gap_w'])
    df2 = pd.merge(df2, df_new, on='node', how='outer')
    df2.to_csv('./Dataset/Graph/node_feature_copy.csv')
    end = time.perf_counter()
    print('Total Time: %s' % (end - start))


def feature_extract_gas1():
    start = time.perf_counter()
    df1 = pd.read_csv('./Dataset/Graph/ETH.csv')
    df1 = df1.sort_values(by=['From', 'GasPrice']).reset_index().drop(columns=['index', 'Unnamed: 0'])
    df2 = pd.read_csv('./Dataset/Graph/node_feature.csv')
    df2 = df2.drop(columns=['Unnamed: 0'])
    i = 0
    arr = [['node', 'min_gasprice_all', 'max_gasprice_all', 'avg_gasprice_all']]
    while i < len(df1.index):
        j = i
        while j < len(df1.index) and df1['From'][j] == df1['From'][i]:
            j += 1
        total_gasprice = 0
        for k in range(i, j):
            total_gasprice += df1['GasPrice'][k]

        min_gas_all = df1['GasPrice'][i]
        max_gas_all = df1['GasPrice'][j - 1]
        avg_gas_all = total_gasprice / (j - i)
        tem_arr = [df1['From'][i], min_gas_all, max_gas_all, avg_gas_all]
        arr.insert(1, tem_arr)
        i = j
        if i % 10000 == 0:
            print('%s data have been processed!' % i)
    arr = np.delete(arr, 0, axis=0)
    df_new = pd.DataFrame(arr, columns=['node', 'min_gasprice_all', 'max_gasprice_all', 'avg_gasprice_all'])
    df2 = pd.merge(df2, df_new, on='node', how='outer')
    df2.to_csv('./Dataset/Graph/node_feature.csv')
    end = time.perf_counter()
    print('Total Time: %s' % (end - start))


def feature_extract_gas2():
    start = time.perf_counter()
    df1 = pd.read_csv('./Dataset/Graph/ETH.csv')
    df1_deposit = df1[df1['Method'] == 'Deposit']
    df1_deposit = df1_deposit.sort_values(by=['From', 'GasPrice']).reset_index().drop(columns=['index', 'Unnamed: 0'])
    df1_withdraw = df1[df1['Method'] == 'Withdraw']
    df1_withdraw = df1_withdraw.sort_values(by=['From', 'GasPrice']).reset_index().drop(columns=['index', 'Unnamed: 0'])
    df2 = pd.read_csv('./Dataset/Graph/node_feature.csv')
    df2 = df2.drop(columns=['Unnamed: 0'])
    i = 0
    arr_deposit = [['node', 'min_gasprice_d', 'max_gasprice_d', 'avg_gasprice_d']]
    arr_withdraw = [['node', 'min_gasprice_w', 'max_gasprice_w', 'avg_gasprice_w']]
    while i < len(df1_withdraw.index):
        j = i
        while j < len(df1_withdraw.index) and df1_withdraw['From'][j] == df1_withdraw['From'][i]:
            j += 1
        total_gasprice = 0
        for k in range(i, j):
            total_gasprice += df1_withdraw['GasPrice'][k]

        min_gas = df1_withdraw['GasPrice'][i]
        max_gas = df1_withdraw['GasPrice'][j - 1]
        avg_gas = total_gasprice / (j - i)
        tem_arr = [df1_withdraw['From'][i], min_gas, max_gas, avg_gas]
        arr_withdraw.insert(1, tem_arr)
        i = j
        if i % 10000 == 0:
            print('%s data have been processed!' % i)
    arr_withdraw = np.delete(arr_withdraw, 0, axis=0)
    df_new = pd.DataFrame(arr_withdraw, columns=['node', 'min_gasprice_w', 'max_gasprice_w', 'avg_gasprice_w'])
    df2 = pd.merge(df2, df_new, on='node', how='outer')
    df2.to_csv('./Dataset/Graph/node_feature.csv')
    end = time.perf_counter()
    print('Total Time: %s' % (end - start))


def normalize_handle():
    start = time.perf_counter()
    df_feature = pd.read_csv('./Dataset/Graph/node_feature.csv')
    df_node = df_feature[['nodeid', 'node']]
    df_feature = df_feature.drop(columns=['nodeid', 'node'])
    df_feature.replace([np.inf, -np.inf], np.nan, inplace=True)
    transfer = preprocessing.StandardScaler().fit_transform(df_feature)
    df_new = pd.DataFrame(transfer, columns=df_feature.columns)
    df_new.insert(0, 'node', df_node['node'])
    df_new.insert(0, 'nodeid', df_node['nodeid'])
    print(df_new)
    df_new.to_csv('./Dataset/Graph/node_feature_normalized.csv')
    end = time.perf_counter()
    print('Total Time: %s' % (end - start))


if __name__ == '__main__':
    # concat_all()
    # concat_gasprice()
    # unique_address()
    # feature_extract_value()
    # feature_extract_time1()
    # feature_extract_time2()
    # feature_extract_gas1()
    # feature_extract_gas2()
    normalize_handle()
    # auto_features()