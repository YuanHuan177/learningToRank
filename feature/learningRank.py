# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  @Time : 2018/11/19 14:29
#  @Author : lg
#  @File : learningRank.py

from sklearn.externals import joblib
import pandas as pd
import numpy as np
from utils import NNfuncs
from utils.evaluation import evalu

np.set_printoptions(threshold=np.inf)  # 让数据全部输出
all_data = joblib.load(r"../data/dealedFeature.jl")
# ['nsrdzdah',1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,27, 28, 29, 30, 31, 32,'XYFZ','WTBZ']

train_data = all_data.dropna(subset=['XYFZ'])
train_feature_data = train_data.drop(columns=['nsrdzdah', 'XYFZ', 'WTBZ'])
train_label_data = train_data['XYFZ'].map(lambda x: 100 - x)  # 标签为信用分
train_label_data2 = train_data['WTBZ']  # 标签为wtbz

print len(all_data[all_data["WTBZ"] == 1.0])  # 有问题 :3729
print len(all_data[all_data["WTBZ"] != 1.0])  # 无问题 :474908
test_data = all_data[all_data['XYFZ'].isnull()]
test_feature_data = test_data.drop(columns=['nsrdzdah', 'XYFZ', 'WTBZ'])

def saveScore(score, path):
    gt_score = pd.concat([test_data[['nsrdzdah', 'WTBZ']], score], axis=1)
    gt_score.to_csv(path, encoding='utf-8', index=False)
    y_pre = gt_score['SCORE'].apply(lambda x: transfer(x))
    evalu(gt_score['WTBZ'], y_pre)


# pointWise 类型
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def transfer(x):
    if x >= 50:
        return 1.0
    else:
        return 0.0


# model = LogisticRegression(random_state=1)
# model.fit(train_feature_data, train_label_data2)
# score = pd.DataFrame(model.predict(test_feature_data), columns=["SCORE"]).set_index(test_feature_data.index)
# saveScore(score, r"../data/compliance1")

# pairWise 类型
# pair_train_feature = pd.concat(
#     [train_feature_data, train_feature_data.shift(-1)], axis=1)[:-1]
# pair_train_label = train_label_data.diff(-1)[:-1]
# model.fit(pair_train_feature, pair_train_label)
# score = pd.DataFrame(model.predict(test_feature_data), columns=["SCORE"])
# saveScore(score, r"../data/compliance2")

from rank import RankNet

rankNetModel = RankNet.RankNet()
rankNetModel.fit(train_feature_data.as_matrix(), train_label_data.as_matrix())
rankNetModel.predict(test_feature_data.as_matrix().astype('float32'))

from regression import NN

NNModel = NN.NN()
NNModel.fit(train_feature_data.as_matrix(), train_label_data.as_matrix())
NNModel.predict(test_feature_data.as_matrix().astype('float32'))

# listWise 类型
from rank import ListNet

listNetModel = ListNet.ListNet()
listNetModel.fit(train_feature_data.as_matrix(), train_label_data.as_matrix())
listNetModel.predict(test_feature_data.as_matrix().astype('float32'))
