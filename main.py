# coding=utf-8
import torch
import random
import os
import time
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import torch.nn.functional as F
from torch_geometric.data import Data
from model import GNN_NET
from data_process import load_10_fold_data
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('Success!')


def mytest(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
        model.train()
    edge_index = data.edge_label_index.cpu().numpy()
    label_test = data.edge_label.cpu().numpy()
    out_np = out.cpu().numpy()
    label_pred = np.where(out_np > 0.75, 1, 0)

    TP, FP, FN, TN = 0, 0, 0, 0
    for i in range(len(label_test)):
        if label_test[i] == 1 and label_pred[i] == 1:
            TP = TP + 1
        elif label_test[i] == 1 and label_pred[i] == 0:
            FN = FN + 1
        elif label_test[i] == 0 and label_pred[i] == 1:
            FP = FP + 1
        else:
            TN = TN + 1

    figure_start_time = time.perf_counter()
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    accuracy = sm.accuracy_score(label_test, label_pred)
    precision = sm.precision_score(label_test, label_pred)
    recall = sm.recall_score(label_test, label_pred)
    f_1 = sm.f1_score(label_test, label_pred)
    roc_auc = sm.roc_auc_score(label_test, out_np)
    figure_end_time = time.perf_counter()
    figure_time = figure_end_time - figure_start_time
    return edge_index, out_np, accuracy, precision, recall, f_1, FPR, FNR, roc_auc, figure_time


def train_10_fold():
    start = time.perf_counter()
    df_pos_data = pd.read_csv('./Dataset/Graph/train_pos_edge_10fold.csv')
    df_neg_data = pd.read_csv('./Dataset/Graph/train_neg_edge_10fold.csv')
    neg_data_index = np.array(df_neg_data['Unnamed: 0'])
    neg_idx_train = []
    neg_idx_test = []

    arr = [['test_accuracy', 'test_precision', 'test_recall', 'test_f_1', 'test_FPR', 'test_FNR', 'test_auc']]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    skf = KFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (train_neg_idx, test_neg_idx) in enumerate(skf.split(df_neg_data)):
        neg_idx_train.append(train_neg_idx)
        neg_idx_test.append(test_neg_idx)

    for fold, (train_pos_idx, test_pos_idx) in enumerate(skf.split(df_pos_data)):
        print('**' * 10, 'The', fold + 1, 'Fold', 'ing....', '**' * 10)
        train_neg_idx = neg_idx_train[fold]
        df_train_pos = df_pos_data.iloc[train_pos_idx]
        df_train_neg = df_neg_data.iloc[train_neg_idx]
        df_train = df_train_pos.append(df_train_neg).reset_index(drop=True).drop(columns=['Unnamed: 0'])  # 重置索引

        train_data, names = load_10_fold_data(df_train)
        train_data = train_data.cuda()
        print(train_data)

        test_neg_idx = neg_idx_test[fold]
        df_test_pos = df_pos_data.iloc[test_pos_idx]
        df_test_neg = df_neg_data.iloc[test_neg_idx]
        df_test = df_test_pos.append(df_test_neg).reset_index(drop=True).drop(columns=['Unnamed: 0'])  # 重置索引

        test_data, names = load_10_fold_data(df_test)
        test_data = test_data.cuda()
        print(test_data)

        print('Data load succeed!')

        # 每一折都要实例化新的模型
        model = GNN_NET(train_data.num_features, 32, 16).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()
        min_epochs = 10
        best_model = None
        best_epoch = 0
        best_test_accuracy = 0
        best_test_precision = 0
        best_test_recall = 0
        best_test_f_1 = 0
        best_test_FPR = 0
        best_test_FNR = 0
        best_test_auc = 0

        print(next(model.parameters()).is_cuda)
        model.train()
        for epoch in range(1, 101):
            optimizer.zero_grad()
            out = model(train_data.x, train_data.edge_index, train_data.edge_label_index).view(-1)
            loss = criterion(out, train_data.edge_label)
            loss.backward()
            optimizer.step()

            # validation
            test_edge, test_out, test_accuracy, test_precision, test_recall, test_f_1, test_FPR, test_FNR, test_auc, \
            test_figure_time = mytest(model, test_data)

            if epoch > min_epochs and test_f_1 > best_test_f_1:
                best_epoch = epoch
                best_test_accuracy = test_accuracy
                best_test_precision = test_precision
                best_test_recall = test_recall
                best_test_f_1 = test_f_1
                best_test_FPR = test_FPR
                best_test_FNR = test_FNR
                best_test_auc = test_auc

        print('best_epoch {:03d} best_test_f_1 {:.4f}'.format(best_epoch, best_test_f_1))
        tmp_arr = [best_test_accuracy, best_test_precision, best_test_recall, best_test_f_1, best_test_FPR, best_test_FNR, best_test_auc]
        arr.insert(1, tmp_arr)

    arr = np.delete(arr, 0, axis=0)
    df_new = pd.DataFrame(arr, columns=['test_accuracy', 'test_precision', 'test_recall', 'test_f_1', 'test_FPR', 'test_FNR', 'test_auc'])
    print(df_new)

    df_new[['test_accuracy', 'test_precision', 'test_recall', 'test_f_1', 'test_FPR', 'test_FNR', 'test_auc']] = \
        df_new[['test_accuracy', 'test_precision', 'test_recall', 'test_f_1', 'test_FPR', 'test_FNR', 'test_auc']].astype('float')

    print('mean_test_accuracy {} test_precision {} test_recall {} test_f_1 {} test_FPR {} test_FNR {} test_auc {}'
          .format(df_new['test_accuracy'].mean(), df_new['test_precision'].mean(), df_new['test_recall'].mean(),
                  df_new['test_f_1'].mean(), df_new['test_FPR'].mean(), df_new['test_FNR'].mean(), df_new['test_auc'].mean()))
    print('test_accuracy_std {} test_precision_std {} test_recall_std {} test_f_1_std {} test_FPR_std {} test_FNR_std {} test_auc_std {}'
          .format(df_new['test_accuracy'].std(), df_new['test_precision'].std(), df_new['test_recall'].std(),
                  df_new['test_f_1'].std(), df_new['test_FPR'].std(), df_new['test_FNR'].std(), df_new['test_auc'].std()))

    end = time.perf_counter()
    print('Total time cost: %s' % (end - start))

    return df_new


if __name__ == "__main__":
    seed_torch(1029)
    train_10_fold()
