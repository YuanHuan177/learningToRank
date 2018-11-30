# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  @Time : 2018/11/21 14:25
#  @Author : lg
#  @File : evaluation.py
import numpy as np
import pandas as pd
cost = np.array([[0, 0], [0, 0]])
# 定义所有的评价指标
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss, \
    precision_recall_curve, roc_curve, average_precision_score


def evalu(y_test, y_pre):
    # 第1个指标是每个cost矩阵对应的cost
    my_matrix = confusion_matrix(y_test, y_pre)
    print("confusion= \n", my_matrix)
    print('auc = ', roc_auc_score(y_test, y_pre))
    print('precision = ', precision_score(y_test, y_pre))
    print('recall', recall_score(y_test, y_pre, average=None))
    print('f1', f1_score(y_test, y_pre))
    # print(classification_report(y_test, y_pre))
    # 第一个指标是精确度
    # 第二个指标是G-mean
    recall_list = recall_score(y_test, y_pre, average=None)
    print('G-mean = ', G_mean(recall_list))
    print("my cost = ", get_cost(my_matrix, cost))

# 获得评价指标G-mean
def G_mean(lists):
    le = len(lists)
    result = 1
    for i in range(le):
        result = lists[i] * result
    return pow(result, 1 / le)


# 获得评价指标Cost
def get_cost(conf_matrix, cost):
    # print(conf_matrix)
    conf_matrix[0, 0] = 0
    conf_matrix[1, 1] = 0
    print(conf_matrix)
    result = np.matmul(cost.T, conf_matrix)
    # result[0,0]是cost(FN)*pro(FN)
    # result[1,1]是cost(FP)*pro(FP)
    # print("the result is:", result)
    cost_sum = result.sum(axis=0).sum(axis=0)
    print("all cost : ", cost_sum)
    return cost_sum


def draw(methods,y_test, y_pre):
    methods
    for i in range(len(methods)):
        fpr, tpr, thresholds = roc_curve(y_test, y_pre)
        df = pd.DataFrame(np.column_stack((fpr, tpr, thresholds)))
        df.to_csv('../data/graph' + methods[i] + '.csv')