# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  @Time : 2018/12/3 11:10
#  @Author : lg
#  @File : EnsembleLearning.py
#   两层集成学习 TLEL方法

import os
import pandas as pd
import numpy as np
from itertools import chain

from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss, \
    precision_recall_curve, roc_curve, average_precision_score

from sklearn.externals import joblib

# vars = ['f' + str(i) for i in range(648)]
m, n = joblib.load(r"../data/clean_reformat_2014.jl").shape
vars = ['f' + str(i) for i in range(n - 2)]


def loadDat():
    """ Load train + test sets and prep data """
    train = joblib.load(r"../data/clean_reformat_2014.jl")  # clean_reformat_2014是经过som-gmm 筛选的，未经过筛选的是 reformat_2014
    test = joblib.load(r"../data/reformat_2015.jl")
    all_data = joblib.load(r"../data/reformat_all.jl")
    train.columns = ['nsrdzdah'] + vars + ['wtbz']
    test.columns = ['nsrdzdah'] + vars + ['wtbz']
    all_data.columns = ['nsrdzdah'] + vars
    finaltrain = pd.concat([train, test])
    print len(train[train['wtbz'] == 1]), len(train[train['wtbz'] == 0])
    print len(test[test['wtbz'] == 1]), len(test[test['wtbz'] == 0])
    # d = pd.concat([test[test['wtbz'] == 0].sample(2000), test[test['wtbz'] == 1].sample(2000)]).reset_index(drop=True)
    # print d.shape
    # return train, d
    return train, test  # 注意这里时 test 为评价指标  all_data 则为给每个节点得分


def bestF1(truth, pred):
    """ Find the optimal F1 value over a grid of cutoffs """
    bestf1 = 0
    bestcut = 0
    precision = 0
    recall = 0

    # for cutoff in np.arange(0.01, 0.99, 0.01):
    #     tmp = f1_score(truth, pd.Series(pred > cutoff).apply(lambda x: 1 if x else 0))
    #     if tmp > bestf1:
    #         bestf1 = tmp
    #         bestcut = cutoff
    #         precision = precision_score(truth, pd.Series(pred > cutoff).apply(lambda x: 1 if x else 0))
    #         recall = recall_score(truth, pd.Series(pred > cutoff).apply(lambda x: 1 if x else 0))
    cutoff = 0.5
    bestf1 = f1_score(truth, pd.Series(pred > cutoff).apply(lambda x: 1 if x else 0))
    precision = precision_score(truth, pd.Series(pred > cutoff).apply(lambda x: 1 if x else 0))
    recall = recall_score(truth, pd.Series(pred > cutoff).apply(lambda x: 1 if x else 0))

    print "bestcut: " + str(bestcut)
    return bestf1, precision, recall


models = {
    'defaultRF': {'n_estimators': 145, 'max_depth': 18, 'min_samples_split': 4, 'max_features': None},
    'defaultGBM': {'n_estimators': 65, 'max_depth': 6, 'learning_rate': 0.300, 'max_features': 'sqrt'}
}


def runRandomUnderSample(train, test, seed):
    for i in range(5):
        X_resampled, y_resampled =ADASYN().fit_sample(train[vars], train.wtbz)
        print (len(X_resampled))
        trained = pd.DataFrame(np.concatenate((X_resampled, y_resampled.reshape(-1, 1)), axis=1))
        trained.columns = vars + ['wtbz']
        rf1Default = baseModel(
            GradientBoostingClassifier(n_estimators=models['defaultGBM']['n_estimators'],
                                       learning_rate=models['defaultGBM']['learning_rate'],
                                       max_depth=models['defaultGBM']['max_depth'],
                                       max_features=models['defaultGBM']['max_features'], random_state=seed + 29),
            vars, "rfbase" + str(i), trained, test, seed)
    return train, test


def baseModel(model, vars, t, train, test, seed):
    kf = KFold(5, shuffle=True, random_state=seed)
    train[t + 'DefaultPred'] = 0.0
    for tr, val in kf.split(train):
        model.fit(train[vars].iloc[train.index[tr]], train['wtbz'].iloc[train.index[tr]])
        train[t + 'DefaultPred'].iloc[train.index[val]] = model.predict_proba(train[vars].iloc[train.index[val]])[:, 1]
    # model.fit(train[vars], train['wtbz'])
    test[t + 'DefaultPred'] = model.predict_proba(test[vars])[:, 1]
    f1, precision, recall = bestF1(test.wtbz, test[t + 'DefaultPred'])
    result = {'AUC': roc_auc_score(test.wtbz, test[t + 'DefaultPred']), 'F1': f1, 'PRECISION': precision,
              'RECALL': recall}
    print t + " PRECISION:  " + str(np.round(result['PRECISION'], 5))
    print t + " RECALL:  " + str(np.round(result['RECALL'], 5))
    print t + " AUC: " + str(np.round(result['AUC'], 5))
    print t + " F1:  " + str(np.round(result['F1'], 5)) + "\n"
    return result


train, test = loadDat()
# make default models
train, test = runRandomUnderSample(train, test, 5)

# 线性回归获取最佳权值配比
myvars = ['rfbase0DefaultPred', 'rfbase1DefaultPred', 'rfbase2DefaultPred', 'rfbase3DefaultPred', 'rfbase4DefaultPred']
# myvars = ['rfDefaultPred', 'gbmDefaultPred']
seed = 5
t = 'ensemble'
test[t + 'DefaultPred'] = 0.0


def vote(param):
    wtbz1 = 0
    wtbz0 = 0
    for v in myvars:
        if param[v] >= 0.5:
            wtbz1 =wtbz1+ 1
        else:
            wtbz0 =wtbz0+ 1
    result=1.0 if wtbz1 > wtbz0 else 0.0
    return result


test[t + 'DefaultPred'] = test[myvars].apply(vote,axis=1)

f1, precision, recall = bestF1(test.wtbz, test[t + 'DefaultPred'])
result = {'AUC': roc_auc_score(test.wtbz, test[t + 'DefaultPred']), 'F1': f1, 'PRECISION': precision,
          'RECALL': recall}
print t + " PRECISION:  " + str(np.round(result['PRECISION'], 5))
print t + " RECALL:  " + str(np.round(result['RECALL'], 5))
print t + " AUC: " + str(np.round(result['AUC'], 5))
print t + " F1:  " + str(np.round(result['F1'], 5)) + '\n'

# 输出预测概率分值, 注意  不用 输出上面的评价指标
# compliance=pd.merge(test[['nsrdzdah','ensembleDefaultPred']],joblib.load(r"../data/reformat_2015.jl")[['nsrdzdah','WTBZ']],on='nsrdzdah',how='left')
# s = (compliance['ensembleDefaultPred'] - compliance['ensembleDefaultPred'].min())/(compliance['ensembleDefaultPred'].max() - compliance['ensembleDefaultPred'].min())
# newcompliance = compliance.drop(['ensembleDefaultPred'],axis=1) #归一化
# newcompliance['ensembleDefaultPred'] = s
# newcompliance.fillna(-1, inplace=True)
# print len(newcompliance[newcompliance['WTBZ']==1])
# print len(newcompliance[newcompliance['WTBZ']==0])
# newcompliance.to_csv(r"../data/compliance" , encoding='utf-8', index=False)


# 画图
from sklearn.metrics import roc_curve

# methods = ['ensemble', 'lda', 'lr', 'dt', 'knn', 'svc','mlp','bayes']
methods = ['lda', 'lr', 'svc', 'bayes', 'dt', 'knn', 'ensemble']
n = len(methods)
f = lambda x: '%.4f' % x
# 对比方法指标
for i in range(n):
    print '========' + str(methods[i])
    print test[methods[i] + 'DefaultPred'].apply(f)

    cutoff = 0.5
    # cutoff = test[methods[i] + 'DefaultPred'].median()
    f1 = f1_score(test.wtbz, pd.Series(test[methods[i] + 'DefaultPred'] > cutoff).apply(lambda x: 1 if x else 0))
    print '%.3f' % f1
    precision = precision_score(test.wtbz,
                                pd.Series(test[methods[i] + 'DefaultPred'] > cutoff).apply(lambda x: 1 if x else 0))
    print '%.3f' % precision

    fpr, tpr, thresholds = roc_curve(test.wtbz, test[methods[i] + 'DefaultPred'].apply(lambda x: 1 if x > 1 else x))
    print  '%.3f' % np.max(tpr - fpr)

    bs = brier_score_loss(test.wtbz,
                          test[methods[i] + 'DefaultPred'].apply(lambda x: x if x > 0 else 0).apply(
                              lambda x: 1 if x > 1 else x))
    print '%.3f' % bs

    ap = average_precision_score(test.wtbz, test[methods[i] + 'DefaultPred'])
    print '%.3f' % ap

    auc = roc_auc_score(test.wtbz, test[methods[i] + 'DefaultPred'])
    print '%.3f' % auc

    a = test.wtbz[test[methods[i] + 'DefaultPred'] < cutoff]
    b = test[methods[i] + 'DefaultPred'][test[methods[i] + 'DefaultPred'] < cutoff]
    try:
        auc = roc_auc_score(a, b)
    except ValueError:
        pass

    pg = 2 * auc - 1
    print '%.3f' % pg

# roc 曲线

for i in range(n):
    fpr, tpr, thresholds = roc_curve(test.wtbz, test[methods[i] + 'DefaultPred'])
    df = pd.DataFrame(np.column_stack((fpr, tpr, thresholds)))
    df.to_csv('../data/roc_' + methods[i] + '.csv', index=False)
# pr 曲线
from sklearn.metrics import precision_recall_curve

for i in range(n):
    precision, recall, thresholds = precision_recall_curve(test.wtbz, test[methods[i] + 'DefaultPred'])
    df = pd.DataFrame(np.column_stack((precision, recall)))
    df.to_csv('../data/pr_' + methods[i] + '.csv')
