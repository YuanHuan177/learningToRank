# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  @Time : 2018/12/3 11:10
#  @Author : lg
#  @File : EnsembleLearning3.py
#  实验ASCI

import operator
import os
import pandas as pd
import numpy as np
from itertools import chain
from scipy.optimize import minimize
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_score
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
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss, \
    precision_recall_curve, roc_curve, average_precision_score

from sklearn.externals import joblib

# vars = ['f' + str(i) for i in range(648)]
m, n = joblib.load(r"../data/clean_reformat_2014.jl").shape
vars = ['f' + str(i) for i in range(n - 2)]
# myvars = ['rfDefaultPred', 'gbmDefaultPred', 'mlpDefaultPred', 'wtbz']
myvars = ['rfDefaultPred', 'gbmDefaultPred', 'mlpDefaultPred', 'ldaDefault', 'lrDefault', 'dtDefault', 'nbDefault',
          'svcDefault', 'knnDefault', 'wtbz']


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


def defaultModel(model, vars, t, train, test, seed):
    """ Make a model for default """
    kf = KFold(5, shuffle=True, random_state=seed)
    train[t + 'DefaultPred'] = 0.0
    for tr, val in kf.split(train):
        model.fit(train[vars].iloc[train.index[tr]], train['wtbz'].iloc[train.index[tr]])
        train[t + 'DefaultPred'].iloc[train.index[val]] = model.predict_proba(train[vars].iloc[train.index[val]])[:, 1]
    model.fit(train[vars], train['wtbz'])
    joblib.dump(model, '../data/' + str(t) + '.jl')

    test[t + 'DefaultPred'] = model.predict_proba(test[vars])[:, 1]
    f1, precision, recall = bestF1(test.wtbz, test[t + 'DefaultPred'])
    result = {'AUC': roc_auc_score(test.wtbz, test[t + 'DefaultPred']), 'F1': f1, 'PRECISION': precision,
              'RECALL': recall}
    print t + " PRECISION:  " + str(np.round(result['PRECISION'], 5))
    print t + " RECALL:  " + str(np.round(result['RECALL'], 5))
    print t + " AUC: " + str(np.round(result['AUC'], 5))
    print t + " F1:  " + str(np.round(result['F1'], 5)) + "\n"
    return result, [t + 'DefaultPred', model]


models = {
    'defaultRF': {'n_estimators': 145, 'max_depth': 18, 'min_samples_split': 4, 'max_features': None},
    'defaultGBM': {'n_estimators': 65, 'max_depth': 6, 'learning_rate': 0.300, 'max_features': 'sqrt'}
}


def runDefaultModels(train, test, seed):
    """ Run all default models """

    # RF Model
    rfDefault, rfModel = defaultModel(
        RandomForestClassifier(n_estimators=models['defaultRF']['n_estimators'],
                               max_depth=models['defaultRF']['max_depth'],
                               min_samples_split=models['defaultRF']['min_samples_split'],
                               max_features=models['defaultRF']['max_features'], n_jobs=10, random_state=seed + 29),
        vars, "rf", train, test, seed)
    # GBM Model
    gbmDefault, gbmModel = defaultModel(
        GradientBoostingClassifier(n_estimators=models['defaultGBM']['n_estimators'],
                                   learning_rate=models['defaultGBM']['learning_rate'],
                                   max_depth=models['defaultGBM']['max_depth'],
                                   max_features=models['defaultGBM']['max_features'], random_state=seed + 29),
        vars, "gbm", train, test, seed)

    mlpDefault, mlpModel = defaultModel(MLPClassifier(), vars, "mlp", train, test, seed)
    ldaDefault = defaultModel(LinearDiscriminantAnalysis(), vars, "lda", train, test, seed)
    lrDefault = defaultModel(LogisticRegression(), vars, "lr", train, test, seed)
    dtDefault = defaultModel(DecisionTreeClassifier(max_depth=18, max_features='sqrt'), vars, "dt", train, test, seed)
    nbDefault = defaultModel(BernoulliNB(), vars, "bayes", train, test, seed)
    svcDefault = defaultModel(SVC(probability=True), vars, "svc", train, test, seed)
    knnDefault = defaultModel(KNeighborsClassifier(), vars, "knn", train, test, seed)
    #     train['defaultPred'] = train['lrDefaultPred']
    #     test['defaultPred'] = test['lrDefaultPred']
    #     train['defaultPred'] = train['rfDefaultPred']*0.55 + train['gbmDefaultPred']*0.45
    #     test['defaultPred'] = test['rfDefaultPred']*0.55 + test['gbmDefaultPred']*0.45
    #     print "blended AUC: " + str(np.round(roc_auc_score(train.wtbz, train['defaultPred']),5))
    #     print "blended F1:  " + str(np.round(bestF1(train.wtbz,train['defaultPred']),5))
    #     print "test blended AUC: " + str(np.round(roc_auc_score(test.wtbz, test['defaultPred']),5))
    #     print "test blended F1:  " + str(np.round(bestF1(test.wtbz,test['defaultPred']),5))
    f1Judgedict = {myvars[0]: rfDefault.get('F1'), myvars[1]: gbmDefault.get('F1'), myvars[2]: mlpDefault.get('F1'),
                   myvars[3]: mlpDefault.get('F1'), myvars[4]: mlpDefault.get('F1'), myvars[5]: mlpDefault.get('F1'),
                   myvars[6]: mlpDefault.get('F1'), myvars[7]: mlpDefault.get('F1'), myvars[8]: mlpDefault.get('F1')}
    f1Judge = pd.DataFrame(sorted(f1Judgedict.items(), key=operator.itemgetter(1), reverse=True))[0]
    allmodel = dict([rfModel, gbmModel, mlpModel, ldaDefault, lrDefault, dtDefault, nbDefault, svcDefault, knnDefault])

    return train, test, f1Judge, allmodel


train, test = loadDat()

# make default models
train, test, f1Judge, allmodel = runDefaultModels(train, test, 5)


def vote(x):
    for classifier in f1Judge:
        if (x['wtbz'] == 1 and x[classifier] >= 0.5) or (x['wtbz'] == 0 and x[classifier] < 0.5):  # 分类正确,优先选择F 值大的
            return classifier
    return f1Judge[0]


s = 'selectClassfier'
train[s + 'wtbz'] = train[myvars].apply(vote, axis=1)
model = DecisionTreeClassifier(max_depth=18, max_features='sqrt')
model.fit(train[myvars], train[s + 'wtbz'])
test[s + 'wtbz'] = model.predict(test[myvars])

# 输出训练后集成方法的指标结果
t = 'ensemble'


def pred(x):
    # return joblib.load(r"../data/" + str(x[s + 'wtbz'])[:-11] + ".jl").predict(x[vars].reshape(1, -1))
    return allmodel.get(x[s + 'wtbz']).predict(x[vars].reshape(1, -1))


train[t + 'DefaultPred'] = pd.concat([train[vars], train[s + 'wtbz']], axis=1).apply(pred, axis=1)[s + 'wtbz']
# joblib.dump(model, '../data/' + str(t) + '.jl')
f1, precision, recall = bestF1(train.wtbz, train[t + 'DefaultPred'])
result = {'AUC': roc_auc_score(train.wtbz, train[t + 'DefaultPred']), 'F1': f1, 'PRECISION': precision,
          'RECALL': recall}
print t + " PRECISION:  " + str(np.round(result['PRECISION'], 5))
print t + " RECALL:  " + str(np.round(result['RECALL'], 5))
print t + " AUC: " + str(np.round(result['AUC'], 5))
print t + " F1:  " + str(np.round(result['F1'], 5)) + '\n'

test[t + 'DefaultPred'] = pd.concat([test[vars], test[s + 'wtbz']], axis=1).apply(pred, axis=1)[s + 'wtbz']
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
for i in range(n):
    precision, recall, thresholds = precision_recall_curve(test.wtbz, test[methods[i] + 'DefaultPred'])
    df = pd.DataFrame(np.column_stack((precision, recall)))
    df.to_csv('../data/pr_' + methods[i] + '.csv')
