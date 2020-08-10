#!/usr/bin/env python
# encoding: utf-8

# @author: ba_ding
# @contact: dinglei_1107@outlook.com
# @file: xgb_origin2lr_base.py
# @time: 2020/5/27 2:32 下午

import os
import pickle
import numpy as np
import scipy.sparse
import xgboost as xgb
from trainer.utils import *
from scipy.sparse import hstack
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing.data import OneHotEncoder

# Make sure the demo knows where to load the data.
DATA_DIR = ".."

# simple example
# load file from text file, also binary buffer generated by xgboost
dtrain = xgb.DMatrix(os.path.join(DATA_DIR, 'data', 'xfea', 'train'))
deval = xgb.DMatrix(os.path.join(DATA_DIR, 'data', 'xfea', 'eval'))

total_train = dtrain.num_row()
pos_in_train = sum(dtrain.get_label())
print("train_data_nums={}, pos_nums={}, neg_nums={}".format(total_train, pos_in_train, total_train - pos_in_train))

# specify parameters via map, definition are same as c++ version
"""
Control the balance of positive and negative weights, useful for unbalanced classes. 
A typical value to consider: sum(negative instances) / sum(positive instances). 
See Parameters Tuning for more discussion. Also, see Higgs Kaggle competition demo for examples: R, py1, py2, py3.
"""
trees = 100
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'max_depth': 3,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'min_child_weight': 5,
    'scale_pos_weight': 1.0,
    'eval_metric': list(['auc', 'logloss']),
    'gamma': 0.2,
    'lambda': 1,
    'alpha': 0,
    'eta': 0.1
}

# specify validations set to watch performance
watchlist = [(deval, 'eval'), (dtrain, 'train')]
num_round = trees
bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_round,
    evals=watchlist,
    feval=normalized_entropy
)

# this is prediction
dtest = xgb.DMatrix(os.path.join(DATA_DIR, 'data', 'xfea', 'test'))
total_test = dtest.num_row()
pos_in_test = sum(dtest.get_label())
print("test_data_nums={}, pos_nums={}, neg_nums={}".format(total_test, pos_in_test, total_test - pos_in_test))

preds = bst.predict(dtest)
labels = dtest.get_label()
xgb_auc1 = roc_auc_score(dtest.get_label(), preds)
_, xgb_ne = normalized_entropy(preds, dtest)
_, xgb_lls = log_loss(preds, dtest)
_, xgb_calibration = calibration(preds, dtest)
print('训练Xgb模型 auc: {:.4f}, ne: {:.4f}, logloss: {:.4f}, calibration: {:.4f}'
      .format(xgb_auc1, xgb_ne, xgb_lls, xgb_calibration))
# print("test top5 pred: {}".format([(labels[i], preds[i]) for i in range(5)]))

# onehot
dr_leaves = bst.predict(dtrain, pred_leaf=True)
dt_leaves = bst.predict(dtest, pred_leaf=True)

all_leaves = np.concatenate((dr_leaves, dt_leaves), axis=0)
all_leaves = all_leaves.astype(np.int32)

# print("dtrain leaves index: \n{}".format(dr_leaves[:5]))

xgb_enc = OneHotEncoder()
X_trans = xgb_enc.fit_transform(all_leaves)
# print("X_train top5 onehot encode:")
# for i in range(5):
#     print(X_trans[i].toarray())

(train_rows, cols) = dr_leaves.shape
print("\nnew x_train shape for lr: {}".format(dr_leaves.shape))
# 定义LR模型
lr = LogisticRegression(max_iter=200)
# lr对xgboost特征编码后的样本模型训练
lr.fit(X_trans[:train_rows, :], dtrain.get_label())
# 预测及AUC评测
y_pred_xgblr1 = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
xgb_lr_auc1 = roc_auc_score(dtest.get_label(), y_pred_xgblr1)
_, xgb_lr_ne = normalized_entropy(y_pred_xgblr1, dtest)
_, xgb_lr_lls = log_loss(y_pred_xgblr1, dtest)
_, xgb_lr_calibration = calibration(y_pred_xgblr1, dtest)
print('训练Xgb-Lr模型 auc: {:.4f}, ne: {:.4f}, logloss: {:.4f}, calibration: {:.4f}'
      .format(xgb_lr_auc1, xgb_lr_ne, xgb_lr_lls, xgb_lr_calibration))

coefs = lr.coef_
intercept = lr.intercept_
lr_result = [",".join([str(x) for x in coefs]), str(intercept)]

# save
bst.save_model(os.path.join(DATA_DIR, 'models', 'xgb_001.model'))
# dump model
bst.dump_model(os.path.join(DATA_DIR, 'models', 'xgb_001.raw.txt'))
# dump model with feature map
# bst.dump_model('../data/test2.nice.txt', os.path.join(DATA_DIR, 'data/featmap.txt'))
write2txt(lr_result, os.path.join(DATA_DIR, 'models', 'lr_001.txt'))