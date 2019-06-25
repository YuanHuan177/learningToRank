# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  @Time : 2018/11/18 11:15
#  @Author : lg
#  @File : cleanData2.py
#   清洗方式：进行kafa检验，互信息
#   √

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
# Data_2014_temp = pd.read_pickle(r"../data2/2014_Data.pickle") # 训练集  7194
# Data14_7194 =pd.read_csv(r"../data2/DATA14_7194.csv",encoding='utf-8', dtype={'nsrdzdah': 'int64'}).drop(['WTBZ'],axis=1)
# Data_2014=pd.merge(Data14_7194,Data_2014_temp)
# Data_2014.to_pickle(r"../data2/2014_Data_7194.pickle")   #训练集   7194
Data_2014 = pd.read_pickle(r"../data2/2014_Data.pickle")   #36944
Data_2015 = pd.read_pickle(r"../data2/2015_Data.pickle")  # 测试集  6663
Data_all = pd.read_pickle(r"../data2/all_Data.pickle")  # 给这些企业得分数  513180

allData_Df = pd.concat([Data_2014[column_order_2014], Data_2015[column_order_2015]])
label = allData_Df['WTBZ']
allDataDf = allData_Df.drop(['WTBZ'], axis=1)

nsrdzdah = allDataDf['nsrdzdah'].reset_index(drop=True)
dataDf = allDataDf.drop(['nsrdzdah'], axis=1)
lossData = pd.DataFrame((dataDf.shape[0] - dataDf.count()) / dataDf.shape[0])
dataDesc = pd.DataFrame(dataDf.describe())
print lossData
print dataDesc
dataDesc.to_csv(r'../data2/dataDesc.csv')

# 2.训练数据 定性指标哑编码 one-hot


DingXingIndex = dataDf[
    ['ZCDZ_YB', 'HY', 'DJZCLX', 'NSRZT', 'NSRLX', 'FDDBR_JG', 'CWFZR_REGION', 'BSR_REGION', 'IS_JC', 'AJLY']]
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
DingXingIndex.fillna(DingXingIndex.mode().T[0], inplace=True)  # 众数补全
ch2, pval = chi2(DingXingIndex, label)  # 卡方检验值统计
for ch in ch2:  # 卡方值
    print 'k %.2f' % ch
for val in pval:  # p值
    print 'p %.2f' % val
KaFang = SelectKBest(chi2, k=9).fit_transform(DingXingIndex, label)
select1 = SelectKBest(chi2, k=9).fit(DingXingIndex, label).get_support(indices=True)
# =====================
allData1415Df = pd.concat([Data_2014[column_order_2014].drop(['WTBZ'], axis=1), Data_all[column_order_all]])
nsrdzdah1415 = allData1415Df['nsrdzdah'].reset_index(drop=True)
data1415Df = allData1415Df.drop(['nsrdzdah'], axis=1)
DingXingIndex1415 = data1415Df[
    ['ZCDZ_YB', 'HY', 'DJZCLX', 'NSRZT', 'NSRLX', 'FDDBR_JG', 'CWFZR_REGION', 'BSR_REGION', 'IS_JC', 'AJLY']]
DingXingIndex1415.fillna(DingXingIndex1415.mode().T[0], inplace=True)
KaFang1415 = DingXingIndex1415.iloc[:, select1]
oneHotXing1415Index = pd.DataFrame(OneHotEncoder().fit_transform(KaFang1415).A)

# =====================
oneHotXingIndex = oneHotXing1415Index.head(len(allData_Df))
mutualIndexXing = SelectKBest(mutual_info_classif, k=20).fit_transform(oneHotXingIndex, label)
select2 = SelectKBest(mutual_info_classif, k=20).fit(oneHotXingIndex, label).get_support(indices=True)

# 3.定量指标 标准化
DingLiangIndex = dataDf.drop(
    ['ZCDZ_YB', 'HY', 'DJZCLX', 'NSRZT', 'NSRLX', 'FDDBR_JG', 'CWFZR_REGION', 'BSR_REGION', 'IS_JC', 'AJLY'],
    axis=1)
DingLiangIndex.fillna(DingLiangIndex.mean(), inplace=True)  # 用平均值补全
StandardIndex = StandardScaler().fit_transform(DingLiangIndex)  # 正态分布标准化
mutualIndexLiang = SelectKBest(mutual_info_classif, k=20).fit_transform(StandardIndex, label)  # 互信息法特征选择
select3 = SelectKBest(mutual_info_classif, k=20).fit(StandardIndex, label).get_support(indices=True)
mi = mutual_info_classif(DingLiangIndex, label)  # 互信息法特征 统计
for m in mi:
    print ('m %.3f' % m)

# 4. 分开测试训练集 备份特征
dealFeature = pd.concat([nsrdzdah, pd.DataFrame(mutualIndexLiang), pd.DataFrame(mutualIndexXing)],
                        axis=1)  # 备份标准化后的数据，todense()是矩阵化
reformat_2014 = pd.concat([dealFeature.head(len(Data_2014)), Data_2014['WTBZ']], axis=1)
reformat_2015 = pd.merge(dealFeature.tail(len(Data_2015)), Data_2015[['nsrdzdah', 'WTBZ']], on='nsrdzdah')
print reformat_2014
print reformat_2015
print dealFeature
print reformat_2014.shape  # 5911*649
print reformat_2015.shape  # 6663*649

joblib.dump(reformat_2014, r"../data2/reformat_2014.jl")  # 4 只做定性 编码，定量数据补全  # 5. 在4基础上，用0补全
joblib.dump(reformat_2015, r"../data2/reformat_2015.jl")

# 对2015年全量数据依照上述方式做特征变换
all_nsrdzdah = Data_all['nsrdzdah'].reset_index(drop=True)
# all_DingXingIndex = Data_all[
#     ['ZCDZ_YB', 'HY', 'DJZCLX', 'NSRZT', 'NSRLX', 'FDDBR_JG', 'CWFZR_REGION', 'BSR_REGION', 'IS_JC', 'AJLY']]
# all_DingXingIndex.fillna(0, inplace=True)
# all_KaFang=all_DingXingIndex.iloc[:,select1]
all_oneHotXingIndex = oneHotXing1415Index.tail(len(Data_all))
oneHotXing1415Index.tail(len(Data_all))
all_mutualIndexXing = all_oneHotXingIndex.iloc[:, select2]

all_DingLiangIndex = Data_all.drop(
    ['ZCDZ_YB', 'HY', 'DJZCLX', 'NSRZT', 'NSRLX', 'FDDBR_JG', 'CWFZR_REGION', 'BSR_REGION', 'IS_JC', 'AJLY'],
    axis=1)
all_DingLiangIndex.fillna(all_DingLiangIndex.mean(), inplace=True)
all_StandardIndex = StandardScaler().fit_transform(all_DingLiangIndex)
all_mutualIndexLiang = pd.DataFrame(all_StandardIndex).iloc[:, select3]

reformat_all = pd.concat([all_nsrdzdah, all_mutualIndexLiang, pd.DataFrame(all_mutualIndexXing)],
                         axis=1)

print reformat_all.shape  # 513180*648
joblib.dump(reformat_all, r"../data2/reformat_all.jl")
