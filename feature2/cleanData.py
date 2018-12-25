# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  @Time : 2018/11/18 11:15
#  @Author : lg
#  @File : cleanData.py

import time

import cx_Oracle
import pandas as pd
import numpy as np
import pandas.io.sql as psql

from AllSql_2014 import table_list_2014, column_order_2014
from AllSql_2015 import table_list_2015, column_order_2015
from AllSql_2015 import table_list_all, column_order_all

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

# 1.数据预处理并进行指标统计
Data_2014 = pd.read_pickle(r"../data/2014_Data.pickle")  # 训练集  5911
Data_2015 = pd.read_pickle(r"../data/2015_Data.pickle")  # 测试集  3086
Data_all = pd.read_pickle(r"../data/all_Data.pickle")  # 给这些企业得分数  513180

allDataDf = pd.concat([Data_2014[column_order_2014].drop(['WTBZ'], axis=1),Data_all[column_order_all]])
print allDataDf.shape

nsrdzdah = allDataDf['nsrdzdah']

# allDataDf.set_index('nsrdzdah', inplace=True)
lossData = pd.DataFrame((allDataDf.shape[0] - allDataDf.count()) / allDataDf.shape[0])
dataDesc = pd.DataFrame(allDataDf.describe())
print lossData
print dataDesc
# dataDesc.to_csv(r"../data/dataDesc.csv", index=False, encoding="utf-8-sig")
# lossData.to_csv(r"../data/lossData.csv", index=False, encoding="utf-8-sig")


# 2.训练数据 定性指标哑编码 one-hot
dataDf = allDataDf
label = dataDf['WTBZ']
DingXingIndex = dataDf[
    ['ZCDZ_YB', 'HY', 'DJZCLX', 'NSRZT', 'NSRLX', 'FDDBR_JG', 'CWFZR_REGION', 'BSR_REGION', 'IS_JC', 'AJLY']]
# 统计 定性指标类别数目
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

ch2, pval = chi2(DingXingIndex, label)  # 卡方检验值统计
for ch in ch2:  # 卡方值
    print 'k %.2f' % ch
for val in pval:  # p值
    print 'p %.2f' % val
KaFang = SelectKBest(chi2, k=10).fit_transform(DingXingIndex, label)  # 进行卡方检验
oneHotXingIndex = OneHotEncoder().fit_transform(KaFang)  # 进行one-hot 编码
print 'oneHotXingIndex shape:' + str(oneHotXingIndex.shape)
mutualIndexXing = SelectKBest(mutual_info_classif, k=14).fit_transform(oneHotXingIndex, label)  # 互信息法特征选择

# 3.定量指标 标准化
DingLiangIndex = dataDf.drop(
    ['ZCDZ_YB', 'HY', 'DJZCLX', 'NSRZT', 'NSRLX', 'FDDBR_JG', 'CWFZR_REGION', 'BSR_REGION', 'IS_JC', 'AJLY'],
    axis=1)
DingLiangIndex.fillna(0, inplace=True)  # 用平均值补全
StandardIndex = StandardScaler().fit_transform(DingLiangIndex)  # 正态分布标准化

mi = mutual_info_classif(DingLiangIndex, label)  # 互信息法特征 统计
for m in mi:
    print ('m %.3f' % m)
mutualIndexLiang = SelectKBest(mutual_info_classif, k=18).fit_transform(StandardIndex, label)  # 互信息法特征选择

# 4. 备份特征
dealFeature = pd.concat(
    [nsrdzdah, pd.DataFrame(mutualIndexLiang), pd.DataFrame(mutualIndexXing.todense()), allDataDf['XYFZ'], label],
    axis=1)  # 备份标准化后的数据，todense()是矩阵化
dealFeature.columns = ['nsrdzdah', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                       24, 25, 26,
                       27, 28, 29, 30, 31, 32, 'XYFZ', 'WTBZ']
print dealFeature.shape
print dealFeature
joblib.dump(dealFeature, r"../data/dealedFeature3.jl")  # 3是用0补全，没有是众数和均值补全

# print dealFeature.shape
# print allDataDf['XYFZ']
# print label
#
# dealedFeaturestemp = pd.concat([allDataDf['XYFZ'],label], axis=1).reset_index(drop=True)
# dealedFeatures= dealFeature
# print dealedFeatures
# dealedFeatures.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
#                           27, 28, 29, 30, 31, 32,'XYFZ','WTBZ']
# dealedFeatures['nsrdzdah']=dealedFeatures.index
# dealedFeatures.reset_index(drop=True)
# print dealedFeatures
# print dealedFeatures.shape

# dealedFeatures.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
#                           27, 28, 29, 30, 31, 32, 'XYFZ']
