#!/usr/bin/env python
# encoding: utf-8

# @author: ba_ding
# @contact: dinglei_1107@outlook.com
# @file: xgb_sklearn2lr_baseline.py
# @time: 2020/5/27 6:21 下午

import numpy as np
import xgboost as xgb
from scipy.sparse import hstack
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing.data import OneHotEncoder

from trainer.utils import print_example4cpp


def xgboost_lr_train(libsvmTrain, libsvmTest):

    # 训练/测试数据分割
    X_train, y_train = load_svmlight_file(libsvmTrain, n_features=127, offset=1)
    X_test, y_test = load_svmlight_file(libsvmTest, n_features=127, offset=1)
    # X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
    print("train top5 example:\n")
    print_example4cpp(X_train, 5, 0)

    print("test top5 example:\n")
    print_example4cpp(X_test, 5, 0)

    # 定义xgb模型
    # nthread = 4, gamma = 0, subsample = 0.9, colsample_bytree = 0.5,
    xgboost = xgb.XGBClassifier(learning_rate=1,
                                n_estimators=2, max_depth=3, missing=-999)

    # xgboost = xgb.XGBRegressor(nthread=4, learning_rate=1,
    #                            n_estimators=2, max_depth=3, gamma=0, subsample=0.9, colsample_bytree=0.5)

    # 训练xgb学习
    xgboost.fit(X_train, y_train)
    xgboost.save_model("../data/test1.model")

    cp_model = xgb.Booster(model_file='../data/test1.model')
    cp_model.dump_model("../data/test1.raw.txt")

    # 预测xgb及AUC评测
    y_pred_test1 = xgboost.predict_proba(X_test)[:, 1]     # for classifier
    # y_pred_test2 = xgboost.predict(X_test)
    xgb_test_auc = roc_auc_score(y_test, y_pred_test1)
    print('xgboost test auc: %.5f' % xgb_test_auc)
    print("test top5 pred1: {}".format(list(zip(y_test[:5], y_pred_test1[:5]))))
    # print("test top5 pred2: {}".format(list(zip(y_test[:5], y_pred_test2[:5]))))

    # xgboost编码原有特征
    X_train_leaves = xgboost.apply(X_train)
    X_test_leaves = xgboost.apply(X_test)

    # 合并编码后的训练数据和测试数据
    All_leaves = np.concatenate((X_train_leaves, X_test_leaves), axis=0)
    All_leaves = All_leaves.astype(np.int32)

    print("X_train leaves index: \n{}".format(X_train_leaves[:5]))

    # 对所有特征进行ont-hot编码
    xgbenc = OneHotEncoder()
    X_trans = xgbenc.fit_transform(All_leaves)
    print("X_train top5 onehot encode:")
    for i in range(5):
        print(X_trans[i].toarray())
    (train_rows, cols) = X_train_leaves.shape
    print("\nnew x_train shape for lr: {}".format(X_train_leaves.shape))
    # 定义LR模型
    lr = LogisticRegression()
    # lr对xgboost特征编码后的样本模型训练
    lr.fit(X_trans[:train_rows, :], y_train)
    # 预测及AUC评测
    y_pred_xgblr1 = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
    xgb_lr_auc1 = roc_auc_score(y_test, y_pred_xgblr1)
    print('基于Xgb特征编码后的LR AUC: %.5f' % xgb_lr_auc1)

    # 定义LR模型
    lr = LogisticRegression(n_jobs=-1, max_iter=100)
    # 组合特征
    X_train_ext = hstack([X_trans[:train_rows, :], X_train])
    X_test_ext = hstack([X_trans[train_rows:, :], X_test])

    # lr对组合特征的样本模型训练
    lr.fit(X_train_ext, y_train)

    # 预测及AUC评测
    y_pred_xgblr2 = lr.predict_proba(X_test_ext)[:, 1]
    xgb_lr_auc2 = roc_auc_score(y_test, y_pred_xgblr2)
    print('基于组合特征的LR AUC: %.5f' % xgb_lr_auc2)


if __name__ == '__main__':
    xgboost_lr_train("../data/agaricus.txt.train", "../data/agaricus.txt.test")
