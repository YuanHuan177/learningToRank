# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  @Time : 2018/11/19 14:29
#  @Author : lg
#  @File : learningRank.py

from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from utils import NNfuncs
from utils.evaluation import evalu, draw
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
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
from regression import NN
from rank import ListNet
from rank import RankNet
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss, \
    precision_recall_curve, roc_curve, average_precision_score

m, n = joblib.load(r"../data2/clean_reformat_2014.jl").shape
vars = ['f' + str(i) for i in range(n - 2)]


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
    return train, test, all_data  # 注意这里时 test 为评价指标  all_data 则为给每个节点得分


np.set_printoptions(threshold=np.inf)  # 让数据全部输出
train, test, all_data = loadDat()


train_feature_data = train.drop(columns=['nsrdzdah', 'wtbz'])
train_label_data = train['wtbz']
train_label_data = pd.merge(train[['nsrdzdah','wtbz']],pd.read_pickle(r"../data2/2014_Data.pickle")[['nsrdzdah','XYFZ']],on='nsrdzdah')['XYFZ']
test_feature_data = test.drop(columns=['nsrdzdah', 'wtbz'])
test_label_data = test['wtbz']

################# 排序学习

def printIndex(method, wtbz, score):
    print '========' + str(method)
    #cutoff = 0.5
    cutoff = pd.Series(score).median()
    f1 = f1_score(wtbz, pd.Series(score > cutoff).apply(lambda x: 1 if x else 0))
    print 'f1: %.3f' % f1
    precision = precision_score(wtbz,
                                pd.Series(score > cutoff).apply(lambda x: 1 if x else 0))
    print 'precision: %.3f' % precision

    fpr, tpr, thresholds = roc_curve(wtbz, pd.Series(score).apply(lambda x: 1 if x > 1 else x))
    print  '%.3f' % np.max(tpr - fpr)

    bs = brier_score_loss(wtbz,
                          pd.Series(score).apply(lambda x: x if x > 0 else 0).apply(
                              lambda x: 1 if x > 1 else x))
    print 'bs: %.3f' % bs

    ap = average_precision_score(wtbz, score)
    print 'ap: %.3f' % ap

    auc = roc_auc_score(wtbz, score)
    print 'auc: %.3f' % auc

    a = test.wtbz[score < cutoff]
    b = score[score < cutoff]
    try:
        auc = roc_auc_score(a, b)
    except ValueError:
        pass

    pg = 2 * auc - 1
    print 'pg: %.3f' % pg


# pairWise 类型
# pair_train_feature = pd.concat(
#     [train_feature_data, train_feature_data.shift(-1)], axis=1)[:-1]
# pair_train_label = train_label_data.diff(-1)[:-1]
# model = SVC(kernel='rbf', probability=True)
# model.fit(pair_train_feature, pair_train_label)
# score = pd.DataFrame(model.predict(test_feature_data), columns=["SCORE"])
# printIndex("PAIR", test_label_data, score)

NNModel = NN.NN()
NNModel.fit(train_feature_data.as_matrix(), train_label_data)
NNScore = NNModel.predict(test_feature_data)

printIndex("NN", test_label_data, NNScore)

rankNetModel = RankNet.RankNet()
rankNetModel.fit(train_feature_data.as_matrix(), train_label_data)
rankNetScore = rankNetModel.predict(test_feature_data)
printIndex("rankNet", test_label_data, rankNetScore)

# listNetModel = ListNet.ListNet()
# listNetModel.fit(train_feature_data.as_matrix(), train_label_data)
# ListNetScore = listNetModel.predict(test_feature_data)
# printIndex("ListNet", test_label_data, ListNetScore)
