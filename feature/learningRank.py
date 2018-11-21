# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  @Time : 2018/11/19 14:29
#  @Author : lg
#  @File : learningRank.py

from sklearn.externals import joblib
import pandas as pd
import numpy as np
from utils import NNfuncs
from utils.evaluation import evalua

np.set_printoptions(threshold=np.inf)  # 让数据全部输出
all_data = joblib.load(r"../data/dealedFeatures.jl")
label = joblib.load(r"../data/label.jl")
xyfz = joblib.load(r"../data/xyfz.jl")

dm, dn = all_data.shape

train_data = all_data.dropna(subset=['XYFZ'])
train_feature_data = train_data.drop(['XYFZ'], axis=1).as_matrix()
train_label_data = train_data['XYFZ'].map(lambda x: 100 - x).as_matrix()
test_data = all_data[all_data['XYFZ'].isnull()]
test_feature_data = test_data.drop(['XYFZ'], axis=1).as_matrix()

# pointWise 类型
from sklearn.ensemble import RandomForestClassifier

# model = RandomForestClassifier(n_estimators=10)
# model.fit(train_feature_data, train_label_data)
# gt_score =pd.DataFrame(model.predict(test_feature_data), columns=["SCORE"])
a=pd.concat([xyfz,label],axis=1)
print a.axes
print a.shape
print a

# pd.concat(xyfz,gt_score,label)


# pairWise 类型
# from rank import RankNet
#
# rankNetModel = RankNet.RankNet()
# rankNetModel.fit(train_feature_data, train_label_data)
# y_pred = rankNetModel.predictTargets(test_feature_data.astype('float32'), batchsize=100)
# print y_pred
#
#
# from regression import NN
# NNModel = NN.NN()
# NNModel.fit(train_feature_data, train_label_data)
# print NNModel.predictTargets(test_feature_data,batchsize=100)
#
#
# # listWise 类型
# from rank import ListNet
# listNetModel = ListNet.ListNet()
# listNetModel.fit(train_feature_data, train_label_data)
# listNetModel.predict(test_feature_data)
