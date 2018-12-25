# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  @Time : 2018/11/19 14:29
#  @Author : lg
#  @File : learningRank.py

from sklearn.externals import joblib
import pandas as pd
import numpy as np
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

np.set_printoptions(threshold=np.inf)  # 让数据全部输出
Data_2014 = joblib.load(r"../data/reformat_2014.jl")
Data_2015 = joblib.load(r"../data/reformat_2015.jl")

all_data = joblib.load(r"../data/reformat_all.jl")

train_feature_data = Data_2014.drop(columns=['nsrdzdah', 'WTBZ'])
print train_feature_data
train_label_data = Data_2014['WTBZ']
test_feature_data = Data_2015.drop(columns=['nsrdzdah', 'WTBZ'])
test_label_data = Data_2015['WTBZ']

# ['nsrdzdah',1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,27, 28, 29, 30, 31, 32,'XYFZ','WTBZ']


model = SVC(kernel='rbf', probability=True)
param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001]}
grid_search = GridSearchCV(model, param_grid, n_jobs=10, verbose=1)
grid_search.fit(train_feature_data, train_label_data)
best_parameters = grid_search.best_estimator_.get_params()
for para, val in list(best_parameters.items()):
    print(para, val)


def get_model(lists):
    # model0 = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    # lists.append(model0)
    # model1 = SVC(C=0.001, gamma=0.001, probability=True, kernel="rbf")
    # lists.append(model1)
    # model2 = LogisticRegression(random_state=1)
    # lists.append(model2)
    # model3 = KNeighborsClassifier()
    # lists.append(model3)
    # model4 = tree.DecisionTreeClassifier()
    # lists.append(model4)
    model5 = RandomForestClassifier(n_estimators=10)
    lists.append(model5)
    return lists


models = []
models = get_model(models)
i = 1
for model in models:
    print("==================================%s========================================")
    scoring = ['accuracy', 'recall', 'f1']
    my_cv = StratifiedKFold(n_splits=10, random_state=1)
    scores = cross_validate(model, train_feature_data, train_label_data, scoring=scoring, cv=my_cv,
                            return_train_score=False)
    # 评价
    model.fit(train_feature_data, train_label_data)
    y_pre = model.predict(test_feature_data)
    evalu(test_label_data, y_pre)
    # 得分
    score = pd.DataFrame(model.predict_proba(all_data)[:, 1], columns=["SCORE"]).set_index(
        all_data['nsrdzdah'])
    score.to_csv(r"../data/compliance" + str(i), encoding='utf-8', index=True)

    i = i + 1


################# 集成学习


# eclf = VotingClassifier(estimators=[('mlp', models[0]), ('dt', models[1]), ('rf', models[2])], voting='hard')  # 无权重投票
# for clf, label in zip([models[0], models[1], models[2], eclf], ['mlp', 'dt', 'rf', 'Ensemble']):
#     scores = cross_val_score(clf, train_feature_data, train_label_data, cv=5, scoring='accuracy')
#     print("准确率: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

#################


# pairWise 类型
# pair_train_feature = pd.concat(
#     [train_feature_data, train_feature_data.shift(-1)], axis=1)[:-1]
# pair_train_label = train_label_data.diff(-1)[:-1]
# model.fit(pair_train_feature, pair_train_label)
# score = pd.DataFrame(model.predict(test_feature_data), columns=["SCORE"])
# saveScore(score, r"../data/compliance2")

# from rank import RankNet
#
# rankNetModel = RankNet.RankNet()
# rankNetModel.fit(train_feature_data.as_matrix(), train_label_data.as_matrix())
# rankNetModel.predict(test_feature_data.as_matrix().astype('float32'))
#
# from regression import NN
#
# NNModel = NN.NN()
# NNModel.fit(train_feature_data.as_matrix(), train_label_data.as_matrix())
# NNModel.predict(test_feature_data.as_matrix().astype('float32'))
#
# # listWise 类型
# from rank import ListNet
#
# listNetModel = ListNet.ListNet()
# listNetModel.fit(train_feature_data.as_matrix(), train_label_data.as_matrix())
# listNetModel.predict(test_feature_data.as_matrix().astype('float32'))


def initSplitData(all_data):
    train_data = all_data.dropna(subset=['XYFZ'])
    train_feature_data = train_data.drop(columns=['nsrdzdah', 'XYFZ', 'WTBZ'])
    train_label_data = train_data['XYFZ'].map(lambda x: 100 - x)  # 标签为信用分
    train_label_data2 = train_data['WTBZ']  # 标签为wtbz

    test_data1 = all_data[all_data['XYFZ'].isnull()]
    test_feature_data1 = test_data1.drop(columns=['nsrdzdah', 'XYFZ', 'WTBZ'])
    return train_feature_data, train_label_data, train_label_data2, test_data1, test_feature_data1


def splitTestData(all_data):
    P_Test = all_data[all_data["WTBZ"] == 1.0]
    num = len(P_Test)
    N_Test = all_data[all_data["WTBZ"] == 0].sample(n=num)
    test_data2 = pd.concat([P_Test, N_Test])
    test_feature_data2 = test_data2.drop(columns=['nsrdzdah', 'XYFZ', 'WTBZ'])
    return test_data2, test_feature_data2


def transfer(x):
    if x >= 50:
        return 1.0
    else:
        return 0.0

# pointWise 类型
