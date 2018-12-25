# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  @Time : 2018/12/3 11:10
#  @Author : lg
#  @File : EnsembleLearning.py

import os
import pandas as pd
import numpy as np
from itertools import chain
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

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss, \
    precision_recall_curve, roc_curve, average_precision_score

from sklearn.externals import joblib

#vars = ['f' + str(i) for i in range(648)]
m,n =joblib.load(r"../data2/reformat_2014.jl").shape
vars = ['f' + str(i) for i in range(n-2)]

def loadDat():
    """ Load train + test sets and prep data """
    train = joblib.load(r"../data2/clean_reformat_2014.jl")  # clean_reformat_2014是经过som-gmm 筛选的，未经过筛选的是 reformat_2014
    test = joblib.load(r"../data2/reformat_2015.jl")
    all_data = joblib.load(r"../data2/reformat_all.jl")
    train.columns = ['nsrdzdah'] + vars + ['wtbz']
    test.columns = ['nsrdzdah'] + vars + ['wtbz']
    all_data.columns = ['nsrdzdah'] + vars
    finaltrain = pd.concat([train, test])
    print len(test[test['wtbz'] == 1]), len(test[test['wtbz'] == 0])
    print len(train[train['wtbz'] == 1]), len(train[train['wtbz'] == 0])
    # d = pd.concat([test[test['wtbz'] == 0].sample(2000), test[test['wtbz'] == 1].sample(2000)]).reset_index(drop=True)
    # print d.shape
    # return train, d
    return finaltrain, all_data  # 注意这里时 test 为评价指标  all_data 则为给每个节点得分


def bestF1(truth, pred):
    """ Find the optimal F1 value over a grid of cutoffs """
    bestf1 = 0
    bestcut = 0
    precision = 0
    recall = 0

    for cutoff in np.arange(0.01, 0.99, 0.01):
        tmp = f1_score(truth, pd.Series(pred > cutoff).apply(lambda x: 1 if x else 0))
        if tmp > bestf1:
            bestf1 = tmp
            bestcut = cutoff
            precision = precision_score(truth, pd.Series(pred > cutoff).apply(lambda x: 1 if x else 0))
            recall = recall_score(truth, pd.Series(pred > cutoff).apply(lambda x: 1 if x else 0))
    # cutoff = 0.5
    # bestf1 = f1_score(truth, pd.Series(pred > cutoff).apply(lambda x: 1 if x else 0))
    # precision = precision_score(truth, pd.Series(pred > cutoff).apply(lambda x: 1 if x else 0))
    # recall = recall_score(truth, pd.Series(pred > cutoff).apply(lambda x: 1 if x else 0))

    print "bestcut: " + str(bestcut)
    return bestf1, precision, recall


def defaultModel(model, vars, t, train, test, seed):
    """ Make a model for default """
    kf = KFold(5, shuffle=True, random_state=seed)
    train[t + 'DefaultPred'] = 0.0
    for tr, val in kf.split(train):
        model.fit(train[vars].iloc[train.index[tr]], train['wtbz'].iloc[train.index[tr]])
        train[t + 'DefaultPred'].iloc[train.index[val]] = model.predict_proba(train[vars].iloc[train.index[val]])[:, 1]
    model.fit(train[vars], train['wtbz'])
    joblib.dump(model, '../data2/' + str(t) + '.jl')

    test[t + 'DefaultPred'] = model.predict_proba(test[vars])[:, 1]
    f1, precision, recall = bestF1(train.wtbz, train[t + 'DefaultPred'])
    result = {'AUC': roc_auc_score(train.wtbz, train[t + 'DefaultPred']), 'F1': f1, 'PRECISION': precision,
              'RECALL': recall}
    print t + " PRECISION:  " + str(np.round(result['PRECISION'], 5))
    print t + " RECALL:  " + str(np.round(result['RECALL'], 5))
    print t + " AUC: " + str(np.round(result['AUC'], 5))
    print t + " F1:  " + str(np.round(result['F1'], 5)) + "\n"
    return result


models = {
    'defaultRF': {'n_estimators': 145, 'max_depth': 18, 'min_samples_split': 4, 'max_features': None},
    'defaultGBM': {'n_estimators': 65, 'max_depth': 6, 'learning_rate': 0.300, 'max_features': 'sqrt'}
}


def runDefaultModels(train, test, seed):
    """ Run all default models """
    # RF Model
    rfDefault = defaultModel(
        RandomForestClassifier(n_estimators=models['defaultRF']['n_estimators'],
                               max_depth=models['defaultRF']['max_depth'],
                               min_samples_split=models['defaultRF']['min_samples_split'],
                               max_features=models['defaultRF']['max_features'], n_jobs=10, random_state=seed + 29),
        vars, "rf", train, test, seed)
    # GBM Model
    gbmDefault = defaultModel(
        GradientBoostingClassifier(n_estimators=models['defaultGBM']['n_estimators'],
                                   learning_rate=models['defaultGBM']['learning_rate'],
                                   max_depth=models['defaultGBM']['max_depth'],
                                   max_features=models['defaultGBM']['max_features'], random_state=seed + 29),
        vars, "gbm", train, test, seed)

    mlpDefault = defaultModel(MLPClassifier(), vars, "mlp", train, test, seed)
    # ldaDefault = defaultModel(LinearDiscriminantAnalysis(), vars, "lda", train, test, seed)
    # lrDefault = defaultModel(LogisticRegression(), vars, "lr", train, test, seed)
    # dtDefault = defaultModel(DecisionTreeClassifier(max_depth=18, max_features='sqrt'), vars, "dt", train, test, seed)
    # nbDefault=defaultModel( BernoulliNB(),vars,"bayes",train,test,seed)
    # svcDefault = defaultModel(SVC(probability=True), vars, "svc", train, test, seed)
    # knnDefault = defaultModel(KNeighborsClassifier(), vars, "knn", train, test, seed)
    #     train['defaultPred'] = train['lrDefaultPred']
    #     test['defaultPred'] = test['lrDefaultPred']
    #     train['defaultPred'] = train['rfDefaultPred']*0.55 + train['gbmDefaultPred']*0.45
    #     test['defaultPred'] = test['rfDefaultPred']*0.55 + test['gbmDefaultPred']*0.45
    #     print "blended AUC: " + str(np.round(roc_auc_score(train.wtbz, train['defaultPred']),5))
    #     print "blended F1:  " + str(np.round(bestF1(train.wtbz,train['defaultPred']),5))
    #     print "test blended AUC: " + str(np.round(roc_auc_score(test.wtbz, test['defaultPred']),5))
    #     print "test blended F1:  " + str(np.round(bestF1(test.wtbz,test['defaultPred']),5))
    return train, test


train, test = loadDat()
# make default models
train, test = runDefaultModels(train, test, 5)

# 线性回归获取最佳权值配比
model = LinearRegression()
myvars =  ['rfDefaultPred', 'mlpDefaultPred','gbmDefaultPred']
#myvars = ['rfDefaultPred', 'gbmDefaultPred']
seed = 5
t = 'ensemble'
kf = KFold(5, shuffle=True, random_state=seed)
train[t + 'DefaultPred'] = 0.0
for tr, val in kf.split(train):
    model.fit(train[myvars].iloc[train.index[tr]], train['wtbz'].iloc[train.index[tr]])
    train[t + 'DefaultPred'].iloc[train.index[val]] = model.predict(train[myvars].iloc[train.index[val]])
model.fit(train[myvars], train['wtbz'])
joblib.dump(model, '../data2/' + str(t) + '.jl')
test[t + 'DefaultPred'] = model.predict(test[myvars])
# f1, precision, recall = bestF1(test.wtbz, test[t + 'DefaultPred'])
# result = {'AUC': roc_auc_score(test.wtbz, test[t + 'DefaultPred']), 'F1': f1, 'PRECISION': precision,
#           'RECALL': recall}
# print t + " PRECISION:  " + str(np.round(result['PRECISION'], 5))
# print t + " RECALL:  " + str(np.round(result['RECALL'], 5))
# print t + " AUC: " + str(np.round(result['AUC'], 5))
# print t + " F1:  " + str(np.round(result['F1'], 5)) + '\n'

# 输出预测概率分值, 注意  不用 输出上面的评价指标
compliance=pd.merge(test[['nsrdzdah','ensembleDefaultPred']],joblib.load(r"../data2/reformat_2015.jl")[['nsrdzdah','WTBZ']],on='nsrdzdah',how='left')
s = (compliance['ensembleDefaultPred'] - compliance['ensembleDefaultPred'].min())/(compliance['ensembleDefaultPred'].max() - compliance['ensembleDefaultPred'].min())
newcompliance = compliance.drop(['ensembleDefaultPred'],axis=1) #归一化
newcompliance['ensembleDefaultPred'] = s
newcompliance.fillna(-1, inplace=True)
print len(newcompliance[newcompliance['WTBZ']==1])
print len(newcompliance[newcompliance['WTBZ']==0])
newcompliance.to_csv(r"../data2/compliance" , encoding='utf-8', index=False)



# 画图
from sklearn.metrics import roc_curve


#methods = ['ensemble', 'lda', 'lr', 'dt', 'knn', 'svc','mlp','bayes']
methods = [ 'lda', 'lr', 'svc','bayes','dt','knn','ensemble']
n = len(methods)
# 对比方法指标
for i in range(n):
    print '========' + str(methods[i])
    #cutoff = 0.5
    cutoff = test[methods[i] + 'DefaultPred'].median()
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
    df.to_csv('../data2/roc_' + methods[i] + '.csv', index=False)
# pr 曲线
from sklearn.metrics import precision_recall_curve

for i in range(n):
    precision, recall, thresholds = precision_recall_curve(test.wtbz, test[methods[i] + 'DefaultPred'])
    df = pd.DataFrame(np.column_stack((precision, recall)))
    df.to_csv('../data2/pr_' + methods[i] + '.csv')

