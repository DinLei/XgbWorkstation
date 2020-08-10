#!/usr/bin/env python
# encoding: utf-8

# @author: ba_ding
# @contact: dinglei_1107@outlook.com
# @file: ctr_feat_reform.py
# @time: 2020/8/10 16:24

"""
摘自：广告CTR预估中真实CTR的离散化方法（https://zhuanlan.zhihu.com/p/34703888）
response_rate往往聚集在头部区间，长尾效应明显
这维特征虽然有用，但作用还不够：首先，对于点击率差距较小的广告，它无法捕捉到相似性；
其次，对于点击率差距较大的广告，也很难捕捉到差异性。这是因为一维数值特征的表达力欠缺引起的。
"""

import math


def entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return - p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)


def calc_gain(labels, indices):
    max_index = max(indices)

    index_map = [[0, 0] for x in range(max_index + 1)]
    i = 0
    for index in indices:
        if labels[i] == 1:
            index_map[index][0] += 1
        else:
            index_map[index][1] += 1
        i += 1
    index_map = filter(lambda t: t[0] > 0 or t[1] > 0, index_map)

    # print "index map:", index_map

    total_positive_count = 0.0
    total_count = 0.0
    for x in index_map:
        total_positive_count += x[0]
        total_count += (x[0] + x[1])

    total_entropy = entropy(total_positive_count / total_count)

    conditional_entropy = 0.0
    for x in index_map:
        index_count = float(x[0] + x[1])
        conditional_entropy += (index_count / total_count) * entropy(x[0] / index_count)

    return total_entropy - conditional_entropy


def domain_transform(ctr, func):
    a = 0.002
    b = 0.3
    ctr = max(a, ctr)
    ctr = min(b, ctr)
    ctr = (ctr - a) / (b - a)
    fa, fb = func(0), func(1)
    assert fa >= 0
    return (func(ctr) - fa) / (fb - fa)


def map_func1(ctrs):
    return [domain_transform(ctr, lambda x: x) for ctr in ctrs]


def map_func2(ctrs):
    return [domain_transform(ctr, lambda x: math.log(1 + 100 * x)) for ctr in ctrs]


def map_func3(ctrs):
    return [domain_transform(ctr, lambda x: math.sqrt(x)) for ctr in ctrs]


def map_func4(ctrs):
    return [mat2gray(ctr) for ctr in ctrs]


def mat2gray(x):
    return x / (x + 0.01)
