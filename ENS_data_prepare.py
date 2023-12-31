# coding=utf-8
import pandas as pd
import numpy as np
import time
import os


def data_clean_ens(filename):
    df = pd.read_csv('./Dataset/Ens/' + filename, usecols=[0, 2, 4, 5, 6, 12, 13])
    error_indexes = df[df['IsError'] == 1].index.tolist()
    error_count = df.loc[error_indexes, 'IsError'].count()
    print(error_count)

    df_new = df.drop(df[df['IsError'] == 1].index)
    df_new = df_new.reset_index()
    df_new = df_new.drop(columns=['index', 'IsError'])
    # df_new.to_csv('./Dataset/Ens/Eth Name Service.csv')
    df_new.to_csv('./Dataset/Ens/Registry with Fallback.csv')


def extract_ens_transfer():
    start = time.perf_counter()
    df1 = pd.read_csv('./Dataset/Ens/Eth Name Service.csv')
    df2 = pd.read_csv('./Dataset/Ens/Registry with Fallback.csv')
    df_transfer1 = df1[df1['Method'] == 'setOwner(bytes32 node, address owner)']
    df_transfer2 = df2[df2['Method'] == 'setOwner(bytes32 node, address owner)']
    df_transfer1 = df_transfer1.append(df_transfer2).drop_duplicates(subset='Txhash')
    df_transfer1 = df_transfer1.sort_values(by=['TimeStamp']).reset_index().drop(columns=['index', 'Unnamed: 0'])
    print(df_transfer1)
    df_transfer1.to_csv('./Dataset/Ens/Transfer.csv')
    end = time.perf_counter()
    print('Total Time: %s' % (end - start))


def extract_ens_newowner():
    start = time.perf_counter()
    df1 = pd.read_csv('./Dataset/Ens/Eth Name Service.csv')
    df2 = pd.read_csv('./Dataset/Ens/Registry with Fallback.csv')
    df_newowner1 = df1[df1['Method'] == 'setSubnodeOwner(bytes32 node, bytes32 label, address owner)']
    df_newowner2 = df2[df2['Method'] == 'setSubnodeOwner(bytes32 node, bytes32 label, address owner)']
    df_newowner1 = df_newowner1.append(df_newowner2).drop_duplicates(subset='Txhash')
    df_newowner1 = df_newowner1.sort_values(by=['TimeStamp']).reset_index().drop(columns=['index', 'Unnamed: 0'])
    print(df_newowner1)
    df_newowner1.to_csv('./Dataset/Ens/NewOwner.csv')
    end = time.perf_counter()
    print('Total Time: %s' % (end - start))


def name_transfer():
    start = time.perf_counter()
    df = pd.read_csv('./Dataset/Ens/Transfer.csv')
    df.insert(2, 'DateTime', 0)
    df['DateTime'] = pd.to_datetime(df['TimeStamp'], unit='s', utc=True)
    i = 0
    drop_list = []
    while i < len(df.index):
        if df['From'][i] == df['New'][i]:
            print(i, df['From'][i], df['New'][i])
            drop_list.append(i)
            i += 1
        else:
            i += 1
    print(df)
    print(len(drop_list))
    df = df.drop(drop_list)
    print(df)
    df = df.drop_duplicates(subset='From', keep=False).reset_index().drop(columns=['index', 'Unnamed: 0'])
    print(df)
    df.to_csv('./Dataset/Ens/Transfer_OnlyOne.csv')
    end = time.perf_counter()
    print('Total Time: %s' % (end - start))


def subdomain_assignment():
    start = time.perf_counter()
    df = pd.read_csv('./Dataset/Ens/NewOwner.csv')
    df.insert(3, 'DateTime', 0)
    df['DateTime'] = pd.to_datetime(df['TimeStamp'], unit='s', utc=True)
    i = 0
    drop_list = []
    while i < len(df.index):
        if df['From'][i] == df['New'][i]:
            print(i, df['From'][i], df['New'][i])
            drop_list.append(i)
            i += 1
        else:
            i += 1
    print(len(drop_list))
    df = df.drop(drop_list)
    print(df)
    df = df.drop_duplicates(subset=['From', 'New']).reset_index().drop(columns=['index', 'Unnamed: 0'])
    print(df)
    df.to_csv('./Dataset/Ens/NewOwner_OnlyOne.csv')
    end = time.perf_counter()
    print('Total Time: %s' % (end - start))


if __name__ == '__main__':
    # data_concat()
    # data_clean_ens('Eth Name Service_NormalTx.csv')
    # data_clean_ens('Registry with Fallback_NormalTx.csv')
    # extract_ens_transfer()
    # extract_ens_newowner()
    # name_transfer()
    subdomain_assignment()