# coding=utf-8
import pandas as pd
import numpy as np
import time
import os


def data_concat():
    file_list = [0.1, 1, 10, 100]
    for val in file_list:
        start = time.perf_counter()
        df_ex = pd.read_csv('./Dataset/External/{}ETH_external.csv'.format(val))
        df_in = pd.read_csv('./Dataset/Internal/{}ETH_internal.csv'.format(val))

        arr_concat = [['Txhash', 'UnixTimestamp', 'DateTime', 'From', 'To', 'Value_IN', 'Value_OUT', 'Method']]
        arr_relayer = [['Txhash', 'DateTime', 'RelayerAddr', 'Value']]

        i, j = 0, 0

        print(len(df_ex))
        for row_ex in df_ex.itertuples():
            if i % 500 == 0:
                print('{} has been processed!'.format(i))
            indexes = df_in[df_in['Txhash'] == row_ex[2]].index.tolist()
            num = df_in.loc[indexes, 'Txhash'].count()

            if num == 2:
                df_temp1 = df_in.loc[indexes[0]]
                df_temp2 = df_in.loc[indexes[1]]

                temp_arr_1 = [row_ex[2], row_ex[3], row_ex[4], df_temp1[7], row_ex[6], row_ex[7], row_ex[8], row_ex[10]]
                arr_concat.insert(1, temp_arr_1)
                temp_arr_2 = [row_ex[2], df_temp2[3], df_temp2[7], df_temp2[9]]
                arr_relayer.insert(1, temp_arr_2)
                df_in = df_in.drop(df_in[df_in['Txhash'] == row_ex[2]].index)
            else:
                temp_arr_3 = [row_ex[2], row_ex[3], row_ex[4], row_ex[5], row_ex[6], row_ex[7], row_ex[8], row_ex[10]]
                arr_concat.insert(1, temp_arr_3)
                df_in = df_in.drop(df_in[df_in['Txhash'] == row_ex[2]].index)

            i += 1
        print("For Loop 1: Success!")

        print(len(df_in))
        for row_in in df_in.itertuples():
            if j % 500 == 0:
                print('{} has been processed!'.format(j))
            indexes = df_in[df_in['Txhash'] == row_in[2]].index.tolist()
            num = df_in.loc[indexes, 'Txhash'].count()

            if num == 2:
                df_temp1 = df_in.loc[indexes[0]]
                df_temp2 = df_in.loc[indexes[1]]

                temp_arr_1 = [row_in[2], row_in[3], row_in[4], df_temp1[7], row_in[7], 0, 0, 'Withdraw']
                arr_concat.insert(1, temp_arr_1)
                temp_arr_2 = [row_in[2], df_temp2[3], df_temp2[7], df_temp2[9]]
                arr_relayer.insert(1, temp_arr_2)
                df_in = df_in.drop(df_in[df_in['Txhash'] == row_in[2]].index)
            else:
                if row_in[10] == val:
                    temp_arr_1 = [row_in[2], row_in[3], row_in[4], row_in[8], row_in[6], 0, 0, 'Withdraw']
                    arr_concat.insert(1, temp_arr_1)
                if row_in[9] == val:
                    temp_arr_1 = [row_in[2], row_in[3], row_in[4], row_in[5], row_in[6], val, 0, 'Deposit']
                    arr_concat.insert(1, temp_arr_1)

            j += 1
        print("For Loop 2: Success!")

        arr_concat = np.delete(arr_concat, 0, axis=0)
        arr_relayer = np.delete(arr_relayer, 0, axis=0)
        df_concat = pd.DataFrame(arr_concat,
            columns=['Txhash', 'UnixTimestamp', 'DateTime', 'From', 'To', 'Value_IN', 'Value_OUT', 'Method'])
        df_relayer = pd.DataFrame(arr_relayer, columns=['Txhash', 'DateTime', 'RelayerAddr', 'Value'])
        df_concat = df_concat.sort_values(by=['UnixTimestamp']).reset_index(drop=True)
        print(df_concat.columns)
        print(df_concat)
        print(df_relayer)
        df_concat.to_csv('./Dataset/Concat/{}ETH-concat.csv'.format(val))
        df_relayer.to_csv('./Dataset/Concat/{}ETH-relayer.csv'.format(val))

        end = time.perf_counter()
        print("Time Cost: %s" % (end - start))


if __name__ == '__main__':
    data_concat()