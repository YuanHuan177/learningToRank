# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  @Time : 2018/11/18 10:10
#  @Author : lg
#  @File : Test.py
import numpy as np
import pandas as pd

neg_table_name = "zk_1212_xkyb"
for fname in [neg_table_name]:
    print fname

y_pred = np.zeros(0)
print y_pred

data=[[8,1,1,0],[3,2,3,2],[6,3,3,3],[9,4,5,4]]
data2=[[8,1,6,0],[3,2,3,2]]
df = pd.DataFrame(data,columns=['q','w','e','r'])
df2 = pd.DataFrame(data2,columns=['q','w','e','r'])
df.set_index('q',inplace=True)
df2.set_index('q',inplace=True)


e=pd.concat(df,df2,on="q")
print e.shape
print e
