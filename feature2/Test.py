# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  @Time : 2018/11/18 10:10
#  @Author : lg
#  @File : Test.py
import numpy as np
import pandas as pd
from sklearn.externals import joblib

# neg_table_name = "zk_1212_xkyb"
# for fname in [neg_table_name]:
#     print fname
#
# y_pred = np.zeros(0)
# print y_pred
#
data1 = pd.DataFrame([[8, 1, 1, 0], [3, 2, 3, 2], [6, 3, 3, 3], [9, 4, 5, 4]], columns=['q', 'w', 'e', 'r'],
                    index=['a', 'b', 'c', 'd'])
data = pd.DataFrame([[8], [3], [6], [9]], columns=['q'],
                    index=['a', 'b', 'c', 'd'])

print data1.r
# print data1
# data2=data1.shift(-1)
# print data2
# print pd.concat([data1.reset_index(drop=True),data1.shift(-1).reset_index(drop=True)],axis=1)[:-1]
# print data.diff(-1)

def transfer(x):
    if x >= 50:
        return 100
    else:
        return 20
score = pd.DataFrame(joblib.load(r"../data/label.jl"))
VAL=score['WTBZ'].apply(lambda x: transfer(x))
print score
print VAL


# print data[0:1]
# data2 = pd.concat([data[1:], data[0:1]])
# print data2.reset_index(drop=True)
# print data.reset_index(drop=True)
# print pd.concat([data.reset_index(drop=True), data2.reset_index(drop=True)], axis=1)
# dataa = data['q'].shift(1)
# print data
# print dataa

# data2=[[8,1,6,0],[3,2,3,2]]
# df = pd.DataFrame(data,columns=['q','w','e','r'])
# df2 = pd.DataFrame(data2,columns=['q','w','e','r'])
# df.set_index('q',inplace=True)
# df2.set_index('q',inplace=True)
#
#
# e=pd.concat(df,df2,on="q")
# print e.shape
# print e

