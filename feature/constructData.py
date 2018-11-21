# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  @Time : 2018/11/16 14:41
#  @Author : lg
#  @File : constructData.py

import time

import cx_Oracle
import pandas as pd
import pandas.io.sql as psql

from AllSql import table_list

conn = cx_Oracle.connect('tax/taxgm2016@192.168.16.186:1521/tax')
print(conn.version)

all_table_prefix = "all_"


# 数据获取
def get_feature(key, conn):
    lowKey = key.lower()
    # 全局变量table_list是dict,   key是当前key
    sentence = table_list[key]
    for fname in [all_table_prefix]:
        sqlline = sentence
        starttime = time.time()
        s1 = time.strftime("%Y-%m-%d-%H-%M-%S")
        print("current table is %s, time: %s" % (key, s1))
        try:
            df = psql.read_sql_query(sqlline, conn)
            outputname = fname + str(lowKey) + ".csv"
            print(outputname, df.shape)
            df.to_csv(r"../data/" + str(outputname), index=False, encoding="utf-8-sig")
        except pd.io.sql.DatabaseError as exc:
            error, = exc.args
            print("Oracle-Error-Message: ", error)
        period = time.time() - starttime
        print("time: %.2f" % period)


def mergeData(tableKeys):
    i = 0
    for t in tableKeys:
        print(all_table_prefix + t.lower() + '.csv')
        df_ = pd.read_csv(r"../data/" + all_table_prefix + t.lower() + '.csv', header=0, names=["nsrdzdah", t],
                          dtype={'nsrdzdah': 'int64', t: 'float64'})
        if (i == 0):
            df = df_
        else:
            df = pd.merge(df, df_, on="nsrdzdah", how="left")
        i += 1
        print(df.shape)
    df.to_pickle(r"../data/allData.pickle")


if __name__ == "__main__":
    conn = cx_Oracle.Connection('tax/taxgm2016@192.168.16.186:1521/tax')
    # 获取特征
    tableKeys = list(table_list.keys())
    for key in tableKeys:
        get_feature(key, conn)
    conn.close()

    # 合并数据
    mergeData(tableKeys)