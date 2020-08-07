#!/usr/bin/env python
# encoding: utf-8

# @author: ba_ding
# @contact: dinglei_1107@outlook.com
# @file: utils.py
# @time: 2020/5/28 11:34 上午

import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# 只给sparse matrix用
def print_example4cpp(data, num, off):
    print("{")
    for i in range(num):
        t_index = data[i].indices
        i2v = list(zip(t_index, [1] * len(t_index)))
        ori_ex = "{"
        for j in range(len(i2v)):
            if j == len(i2v) - 1:
                ori_ex += ("{" + str(i2v[j][0] + off) + "," + str(i2v[j][1]) + "}}")
            else:
                ori_ex += ("{" + str(i2v[j][0] + off) + "," + str(i2v[j][1]) + "},")
        print(ori_ex + "," if i < (num-1) else ori_ex + "\n}")


def xgb_binary2txt(model_file, to_file):
    import xgboost as xgb
    cp_model = xgb.Booster(model_file=model_file)
    cp_model.dump_model(to_file)


def libsvm_check(file):
    keys = set()
    with open(file, "r") as fin:
        for line in fin:
            eles = line.split()
            for ele in eles[1:]:
                keys.add(int(ele.split(":")[0]))
    keys = list(keys)
    keys.sort()
    print(keys)


def write2txt(data, file):
    with open(file, "w") as fin:
        for line in data:
            fin.write(line)
            fin.write("\n")


def calibration(preds, eval_data):
    # import xgboost as xgb
    # assert isinstance(eval_data, xgb.core.DMatrix)
    labels = eval_data.get_label()
    return 'calibration', sum(preds)/sum(labels)


def normalized_entropy(preds, eval_data):
    labels = np.array(eval_data.get_label()).reshape(-1)
    probs = np.array(preds).reshape(-1)
    avg_prob = sum(labels) / len(labels)
    num = (-1.0 / len(labels)) * sum(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    den = -1.0 * (avg_prob * np.log(avg_prob) + (1 - avg_prob) * np.log(1 - avg_prob))
    return 'norm_entropy', num/den


def log_loss(preds, eval_data):
    labels = np.array(eval_data.get_label()).reshape(-1)
    probs = np.array(preds).reshape(-1)
    num = (-1.0 / len(labels)) * sum(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    return 'logloss', num


def xfea_list2fmap(xfea_list, fmap):
    count = 0
    fmaps = []
    with open(xfea_list, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            # name=nativeflag;class=S_onehot;slot=211;depend=nativeflag;is_hash=false;feat_values=0,1,2
            tokens = dict([x.split("=") for x in line.split(";")])
            fname = tokens['name']
            ftype = 'i'
            if 'feat_type' in tokens and tokens['feat_type'] == '0':
                ftype = 'q'
            fvalues = []
            if 'feat_values' in tokens:
                fvalues = tokens['feat_values'].split(",")
            if len(fvalues) <= 1:
                fmaps.append("{}\t{}\t{}\n".format(count, fname, ftype))
                count += 1
            else:
                for cname in fvalues:
                    fmaps.append("{}\t{}={}\t{}\n".format(count, fname, cname, ftype))
                    count += 1
    with open(fmap, "w", encoding="utf8") as fout:
        for line in fmaps:
            fout.write(line)


def get_features_importance(model, fmap="", importance_type='weight', output=""):
    import xgboost as xgb
    booster = xgb.Booster(model_file=model)
    importance = booster.get_score(fmap=fmap, importance_type=importance_type)
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    num = 1
    if output:
        with open(output, "w", encoding="utf8") as fout:
            fout.write("feature_name,importance,order\n")
            for x in importance:
                fout.write("{},{},{}\n".format(x[0], x[1], num))
                num += 1
    # else:
    #     from xgboost import plot_importance
    #     from matplotlib import pyplot
    #     plot_importance(booster)
    #     pyplot.show()
    #     print(importance)


if __name__ == "__main__":
    # import sys
    # mf, of = sys.argv[1:]
    # xgb_binary2txt(mf, of)

    # xfea_list2fmap(
    #     "../models/dinglei_features_list.conf",
    #     "../models/dinglei_features_map.txt")

    get_features_importance(
        model="../models/xgb_001",
        fmap="../models/dinglei_features_map.txt",
        output="../models/dinglei_importance.csv"
    )
    # ibsvm_check("../data/xfea/train/15-part-00190")
