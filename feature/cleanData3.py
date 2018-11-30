# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  @Time : 2018/11/18 11:15
#  @Author : lg
#  @File : cleanData2.py

import time

import cx_Oracle
import pandas as pd
import numpy as np
import pandas.io.sql as psql

from AllSql_2014 import table_list_2014, column_order_2014
from AllSql_2015 import table_list_2015, column_order_2015
from AllSql_all import table_list_all, column_order_all

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

# 1.数据预处理并进行指标统计
Data_2014 = pd.read_pickle(r"../data/2014_Data.pickle")  # 训练集  5911
Data_2015 = pd.read_pickle(r"../data/2015_Data.pickle")  # 测试集  3086
Data_all = pd.read_pickle(r"../data/all_Data.pickle")  # 给这些企业得分数  513657

allDataDf = pd.concat([Data_2014[column_order_2014].drop(['WTBZ'], axis=1), Data_all[column_order_all]])
print allDataDf.shape

nsrdzdah = allDataDf['nsrdzdah'].reset_index(drop=True)
dataDf = allDataDf.drop(['nsrdzdah'], axis=1)
# allDataDf.set_index('nsrdzdah', inplace=True)
lossData = pd.DataFrame((dataDf.shape[0] - dataDf.count()) / dataDf.shape[0])
dataDesc = pd.DataFrame(dataDf.describe())
print lossData
print dataDesc
# dataDesc.to_csv(r"../data/dataDesc.csv", index=False, encoding="utf-8-sig")
# lossData.to_csv(r"../data/lossData.csv", index=False, encoding="utf-8-sig")


# 2.训练数据 定性指标哑编码 one-hot
# label = dataDf['WTBZ']
DingXingIndex = dataDf[
    ['ZCDZ_YB', 'HY', 'DJZCLX', 'NSRZT', 'NSRLX', 'FDDBR_JG', 'CWFZR_REGION', 'BSR_REGION', 'IS_JC', 'AJLY']]
# 统计个定性指标类别数目
print DingXingIndex['ZCDZ_YB'].value_counts().shape
print DingXingIndex['HY'].value_counts().shape
print DingXingIndex['DJZCLX'].value_counts().shape
print DingXingIndex['NSRZT'].value_counts().shape
print DingXingIndex['NSRLX'].value_counts().shape
print DingXingIndex['FDDBR_JG'].value_counts().shape
print DingXingIndex['CWFZR_REGION'].value_counts().shape
print DingXingIndex['BSR_REGION'].value_counts().shape
print DingXingIndex['IS_JC'].value_counts().shape
print DingXingIndex['AJLY'].value_counts().shape
print '定性指标类别数： ' + str(DingXingIndex.describe())
DingXingIndex.fillna(0, inplace=True)  # 用众数在原位置处补全数据
# print XingIndex.columns[np.where(np.isfinite(XingIndex))[0]]  # 判断数据中无穷大在什么位置

oneHotXingIndex = OneHotEncoder().fit_transform(DingXingIndex)  # 进行one-hot 编码
print 'oneHotXingIndex shape:' + str(oneHotXingIndex.shape)
mutualIndexXing = oneHotXingIndex

# 3.定量指标 标准化
DingLiangIndex = dataDf.drop(
    ['ZCDZ_YB', 'HY', 'DJZCLX', 'NSRZT', 'NSRLX', 'FDDBR_JG', 'CWFZR_REGION', 'BSR_REGION', 'IS_JC', 'AJLY'],
    axis=1)
DingLiangIndex.fillna(0, inplace=True)  # 用平均值补全
StandardIndex = StandardScaler().fit_transform(DingLiangIndex)  # 正态分布标准化
mutualIndexLiang = StandardIndex

# 4. 分开测试训练集 备份特征
dealFeature = pd.concat([nsrdzdah, pd.DataFrame(mutualIndexLiang), pd.DataFrame(mutualIndexXing.todense())],
                        axis=1)  # 备份标准化后的数据，todense()是矩阵化
reformat_2014 = pd.concat([dealFeature.head(len(Data_2014)), Data_2014['WTBZ']], axis=1)
reformat_all = dealFeature.tail(len(Data_all))
reformat_2015 = pd.merge(reformat_all,Data_2015[['nsrdzdah','WTBZ']],on='nsrdzdah')

print reformat_2014.axes  # 5911*650
print reformat_all.axes  # 513657*649
print reformat_2015.axes  #3086*650

joblib.dump(reformat_2014, r"../data/reformat_2014.jl")  # 4 只做定性 编码，定量数据补全  #5.在4基础上，用0补全
joblib.dump(reformat_all, r"../data/reformat_all.jl")
joblib.dump(reformat_2015, r"../data/reformat_2015.jl")