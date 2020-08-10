import numpy as np
import xgboost as xgb
from scipy.sparse import hstack
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing.data import OneHotEncoder


def xgboost_lr_train(libsvmTrain, libsvmTest):
    # simple example
    # 训练/测试数据分割
    X_train, y_train = load_svmlight_file(libsvmTrain, n_features=127, offset=1)
    X_test, y_test = load_svmlight_file(libsvmTest, n_features=127, offset=1)
    # X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
