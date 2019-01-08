# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  @Time : 2018/12/4 13:40
#  @Author : lg
#  @File : som_gmm.py

import os
import pandas as pd
import numpy as np
from itertools import chain
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss, \
    precision_recall_curve, roc_curve, average_precision_score
from sklearn.cluster import DBSCAN
from sklearn.externals import joblib
from minisom import MiniSom
from numpy import (array, unravel_index, nditer, linalg, random, subtract,
                   power, exp, pi, zeros, arange, outer, meshgrid, dot, ones)

Data_2014 = joblib.load(r"../data2/reformat_2014.jl")
X = joblib.load(r"../data2/reformat_2014.jl").drop(columns=['nsrdzdah', 'WTBZ'])
Y = joblib.load(r"../data2/reformat_2014.jl")['WTBZ']
nsr = Data_2014['nsrdzdah']
dm, dn = X.shape

som = MiniSom(20, 20, dn, sigma=1.0, learning_rate=1)  # initialization of 6x6 SOM
som.train_random(X.values, dm)  # trains the SOM with 100 iterations
joblib.dump(som.get_weights(), '../data2/som20*20.jl')

som = MiniSom(20, 20, dn, sigma=1.0, learning_rate=1)
som_weights = joblib.load('../data2/som20*20.jl')
dism = som.distance_map()
t = X.values[Y.values == 1, :]
mapcount2 = som.activation_response(t)
mapcount = som.activation_response(X.values)

pd.DataFrame(mapcount).to_csv('../data2/allcount_before.csv')  # 类内样本数量
pd.DataFrame(mapcount2 / mapcount).to_csv('../data2/error_before.csv')  # 分类标签分歧
pd.DataFrame(dism).to_csv('../data2/dism_before.csv')  # 类间距离
allcount = pd.read_csv('../data2/allcount_before.csv', header=0, index_col=0).values
err = pd.read_csv('../data2/error_before.csv', header=0, index_col=0).values
dism = pd.read_csv('../data2/dism_before.csv', header=0, index_col=0).values
from sklearn.preprocessing import MinMaxScaler
d=pd.DataFrame(MinMaxScaler().fit_transform(allcount))
d.to_csv('../data2/allcount_before1.csv')
# print allcount
e=pd.DataFrame(err)
e.to_csv('../data2/error_before1.csv')


# mask决定该网络节点是否成立
mr, mc, me = som_weights.shape
mask = ones((mr, mc))
for i in range(mr):
    for j in range(mc):
        if allcount[i, j] == 0 or (allcount[i, j] < 50 and dism[i, j] > 0.2 and err[i, j] < 0.1):
            mask[i, j] = 0
mask = mask.astype('int8')
XY_mask = [i for i, x in enumerate(X.values) if mask[som.winner(x)] == 1]  # 保留下来数据的下标行数
joblib.dump(X.values[XY_mask], '../data2/filtered_inputFeatures.jl')
joblib.dump(Y.values[XY_mask], '../data2/filtered_categoryLabel.jl')
joblib.dump(nsr.values[XY_mask],'../data2/filtered_nsr.jl')

from collections import defaultdict

coreindex = defaultdict(list)  # 初始化字典，类型为list
passed = mask == 0  # true /false
directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]


def find(x, y, passed, area):
    passed[x, y] = True
    area.append([x, y])
    for dx, dy in directions:
        if (i + dx >= 0 and i + dx < mr and j + dy >= 0 and j + dy < mc and (not passed[i + dx, j + dy]) and dism[
            i, j] <= 0.13):
            find(i + dx, j + dy, passed, area)
    return area


for i in range(mr):
    for j in range(mc):
        if not passed[i, j]:
            if (dism[i, j] > 0.13):
                coreindex[(i, j)] = [[i, j]]
                passed[i, j] = True
            else:
                find(i, j, passed, coreindex[(i, j)])
cores = {}
for (i, j) in coreindex.keys():
    core = np.zeros(me)
    count = 0
    for index in coreindex[(i, j)]:
        core += som.get_weights()[i, j]
        count += 1
    core /= count
    cores[(i, j)] = core
joblib.dump(cores, '../data2/kmeans-init.jl')





X = joblib.load('../data2/filtered_inputFeatures.jl')
Y = joblib.load('../data2/filtered_categoryLabel.jl')
cores = joblib.load('../data2/kmeans-init.jl')
filter_nsr = joblib.load('../data2/filtered_nsr.jl')
k = len(cores)
dm, dn = X.shape
incores = np.array(cores.values())
n_samples, n_features = X.shape

print allcount
allcount[mask==0] = allcount.min()
print allcount
c=pd.DataFrame(MinMaxScaler().fit_transform(allcount))
print c
c[mask==0]=-1
print c
c.to_csv('../data2/allcount_after1.csv')


err[mask==0] = -1
dism[mask==0] = -1
pd.DataFrame(allcount).to_csv('../data2/allcount_after.csv')
pd.DataFrame(err).to_csv('../data2/error_after1.csv')
pd.DataFrame(dism).to_csv('../data2/dism_after.csv')


km = KMeans(init=incores, n_clusters=k, n_init=1)
#km = GaussianMixture(n_components=k,means_init=incores)
km.fit(X)
joblib.dump(km, '../data2/km.jl')


def winner(x):
    pre = km.predict(x.reshape(1, -1))
    return pre[0]


def activation_response(data):
    a = zeros(k)
    for x in data:
        a[winner(x)] += 1
    return a


t = X[Y == 1, :]
abcount = activation_response(t)


allcount = activation_response(X)
err = abcount / allcount
for i, x in enumerate(X):
    ind = winner(x)
    if err[ind] < 0.01 and abcount[ind] < 20:
        Y[i] = 0
clean_reformat_2014 = pd.concat([pd.DataFrame(filter_nsr), pd.DataFrame(X), pd.DataFrame(Y, columns=['WTBZ'])], axis=1)
joblib.dump(clean_reformat_2014, '../data2/clean_reformat_2014.jl')

# c.to_csv('../data2/allcount_before1.csv')
# print allcount
# e=pd.DataFrame(err)
# e.fillna(0,inplace=True)
# e.to_csv('../data2/error_before1.csv')
