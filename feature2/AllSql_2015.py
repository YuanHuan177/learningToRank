# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  @Time : 2018/11/16 14:44
#  @Author : lg
#  @File : AllSql_2015.py

# -*- coding: utf8 -*-
column_order_2015 = ['nsrdzdah', 'ZCDZ_YB', 'LSGX', 'HY', 'DJZCLX', 'KY_MONTH', 'NSRZT', 'NSRLX', 'ZCDNSR_NUM', 'ZRR_NUM',
                'FDDBR_AGE', 'FDDBR_JG', 'CWFZR_AGE', 'CWFZR_REGION', 'BSR_AGE', 'BSR_REGION', 'CYRS', 'TZZE', 'ZCZB',
                'NXXZE', 'XFQC_NUM', 'NGXZE', 'GFQC_NUM', 'SF_RATIO_AVG', 'SF_AVG', 'SJDKSE_RATIO_AVG',
                'YNSE_RATIO_AVG', 'XFJSHJ_MEAN', 'XFJSHJ_MEDIAN', 'IS_JC', 'AJLY', 'XSFYL', 'LRL', 'WTBZ']

table_list_2015 = {
    'ZCDZ_YB':  # 注册地址邮编
        '''
        SELECT
          b.nsrdzdah,
          TO_NUMBER(substr(a.ZCD_YB, 1, 4)) as ZCDZ_YB
        FROM DJ_NSRXX_KZ a RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) b ON a.NSRDZDAH = b.nsrdzdah
        ''',
    'LSGX':  # 隶属关系
        '''
       SELECT
          b.nsrdzdah,
          TO_NUMBER(a.lsgx_dm) as LSGX
        FROM DJ_NSRXX a RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) b ON a.NSRDZDAH = b.nsrdzdah
        ''',
    'HY':  # 行业
        '''
       SELECT
          b.nsrdzdah,
          TO_NUMBER(a.hy_dm) as HY
        FROM DJ_NSRXX a RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) b ON a.NSRDZDAH = b.nsrdzdah
         ''',
    'DJZCLX':  # 登记注册类型
        '''
        SELECT
          b.nsrdzdah,
          TO_NUMBER(a.DJZCLX_DM) as DJZCLX
        FROM DJ_NSRXX a RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) b ON a.NSRDZDAH = b.nsrdzdah
        ''',
    'KY_MONTH':  # 开业至今月数
        '''
        SELECT
          b.nsrdzdah,
          round(months_between(SYSDATE, KYSLRQ)) AS KY_MONTH
        FROM DJ_NSRXX_KZ a RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) b ON a.NSRDZDAH = b.nsrdzdah
        ''',
    'NSRZT':  # 纳税人状态
        '''
        SELECT
          b.nsrdzdah,
          TO_NUMBER(a.NSRZT_dm) as NSRZT
        FROM DJ_NSRXX a RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) b ON a.NSRDZDAH = b.nsrdzdah
        ''',
    'NSRLX':  # 一般纳税人标识
        '''
        SELECT
          b.nsrdzdah,
          (CASE WHEN a.YBNSR_BZ IS NULL THEN 0 ELSE 1 END) as NSRLX
        FROM DJ_NSRXX_KZ a RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) b ON a.NSRDZDAH = b.nsrdzdah
        ''',
    'ZCDNSR_NUM':  # 注册地址的注册纳税人个数
        '''
        SELECT
          q.nsrdzdah,
          p.ZCDNSR_NUM
        FROM
          (SELECT
             NSRDZDAH,
             num_zcdz AS ZCDNSR_NUM
           FROM (SELECT
                   ZCDZ,
                   count(*) AS num_zcdz
                 FROM DJ_NSRXX_KZ
                 GROUP BY ZCDZ) a, DJ_NSRXX_KZ b
           WHERE a.ZCDZ = b.ZCDZ)
          p RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) q ON p.NSRDZDAH = q.nsrdzdah
        ''',
    'ZRR_NUM':  # 关联自然人(投资方、股东)个数
        '''
        SELECT
          b.nsrdzdah,
          TZF+GD AS ZRR_NUM
        FROM 
        (
        SELECT A.NSRDZDAH,TZF,GD
        FROM
        (SELECT
                NSRDZDAH,
                count(DISTINCT (ZJHM)) AS TZF
              FROM DJ_TZF a
              WHERE EXTRACT (YEAR FROM a.LRRQ) <2016
              GROUP BY NSRDZDAH) a 
              full join
          (
            SELECT XH.NSRDZDAH,count(DISTINCT (GD.ZJHM)) AS GD
            FROM SB_NDSDS_2014 xh, SB_NDSDS_2014_JCXX_GD gd 
            WHERE XH.PZXH = GD.PZXH 
            AND   EXTRACT (YEAR FROM xh.sssq_q) < 2016
            AND EXTRACT (YEAR FROM xh.sssq_z) > 2014
            GROUP BY XH.NSRDZDAH
          )b on  a.NSRDZDAH=B.NSRDZDAH) a
        RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) b ON a.NSRDZDAH = b.nsrdzdah       
        ''',
    'FDDBR_AGE':  # 法定代表人年龄
        '''
        SELECT
          q.nsrdzdah,
          p.FDDBRage AS FDDBR_AGE FROM
          ((SELECT
              NSRDZDAH,
              117 - TO_NUMBER(substr(ZJHM, 7, 2)) AS FDDBRage
            FROM DJ_NSRXX
            WHERE REGEXP_LIKE(ZJHM, '^[0-9]{15}$'))
           UNION ALL
           (SELECT
              NSRDZDAH,
              117 - TO_NUMBER(substr(ZJHM, 9, 2)) AS FDDBRage
            FROM DJ_NSRXX
            WHERE REGEXP_LIKE(ZJHM, '^[0-9]{17}([0-9]|X)$')))
          p RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) q ON p.NSRDZDAH = q.nsrdzdah
        ''',
    'FDDBR_JG':  # 法定代表人籍贯
        '''
        SELECT
          b.nsrdzdah,
          substr(ZJHM, 1, 2) AS FDDBR_JG
        FROM (Select * from DJ_NSRXX WHERE REGEXP_LIKE(ZJHM, '^[0-9]{2}.*$')) a RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) b ON a.NSRDZDAH = b.nsrdzdah
        ''',
    'CWFZR_AGE':  # 财务负责人年龄
        '''
        SELECT
          q.nsrdzdah,
          p.CWFZRage AS CWFZR_AGE
        FROM
          ((SELECT
              NSRDZDAH,
              117 - TO_NUMBER(substr(CWFZR_ZJHM, 7, 2)) AS CWFZRage
            FROM DJ_NSRXX_KZ
            WHERE REGEXP_LIKE(CWFZR_ZJHM, '^[0-9]{15}$'))
           UNION ALL
           (SELECT
              NSRDZDAH,
              117 - TO_NUMBER(substr(CWFZR_ZJHM, 9, 2)) AS CWFZRage
            FROM DJ_NSRXX_KZ
            WHERE REGEXP_LIKE(CWFZR_ZJHM, '^[0-9]{17}([0-9]|X)$')))
          p RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) q ON p.NSRDZDAH = q.nsrdzdah
        ''',
    'CWFZR_REGION':  # 财务负责人所属地区
        '''
        SELECT
          b.nsrdzdah,
          substr(a.CWFZR_ZJHM, 1, 2) AS CWFZR_REGION
        FROM (Select * from DJ_NSRXX_KZ WHERE REGEXP_LIKE(CWFZR_ZJHM, '^[0-9]{2}.*$')) a RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) b ON a.NSRDZDAH = b.nsrdzdah
        ''',
    'BSR_AGE':  # 办税人年龄
        '''
        SELECT
          q.nsrdzdah,
          p.BSRage AS BSR_AGE
        FROM
          ((SELECT
              NSRDZDAH,
              117 - TO_NUMBER(substr(BSR_ZJHM, 7, 2)) AS BSRage
            FROM DJ_NSRXX_KZ
            WHERE REGEXP_LIKE(BSR_ZJHM, '^[0-9]{15}$'))
           UNION ALL
           (SELECT
              NSRDZDAH,
              117 - TO_NUMBER(substr(BSR_ZJHM, 9, 2)) AS BSRage
            FROM DJ_NSRXX_KZ
            WHERE REGEXP_LIKE(BSR_ZJHM, '^[0-9]{17}([0-9]|X)$')))
          p RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) q ON p.NSRDZDAH = q.nsrdzdah
        ''',
    'BSR_REGION':  # 办税人所属地区
        '''
        SELECT
          b.nsrdzdah,
          substr(a.BSR_ZJHM, 1, 2) AS BSR_REGION
        FROM (Select * from DJ_NSRXX_KZ WHERE REGEXP_LIKE(BSR_ZJHM, '^[0-9]{2}.*$')) a RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) b ON a.NSRDZDAH = b.nsrdzdah
        ''',
    'CYRS':  # 从业人数
        '''
        SELECT
          b.nsrdzdah,
          CYRS
        FROM DJ_NSRXX_KZ a RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) b ON a.NSRDZDAH = b.nsrdzdah
        ''',
    'TZZE':  # 投资总额
        '''
        SELECT
          b.nsrdzdah,
          a.TZZE
        FROM (SELECT
                NSRDZDAH,
                sum(TZJE) AS TZZE
              FROM DJ_TZF a
              WHERE EXTRACT (YEAR FROM a.LRRQ) <2016
              GROUP BY NSRDZDAH) a RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) b ON a.NSRDZDAH = b.nsrdzdah
        ''',
    'ZCZB':  # 注册资本
        '''
        SELECT
          b.nsrdzdah,
          ZCZB
        FROM DJ_NSRXX_KZ a RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) b ON a.NSRDZDAH = b.nsrdzdah
        ''',
    'NXXZE':  # 年销项总额
        '''
        SELECT
          q.nsrdzdah,
          p.FP_ZZSZYFP_SYQYGS_sum as NXXZE
        FROM
          (SELECT
             b.NSRDZDAH,
             sum(a.FP_ZZSZYFP_SYQYGS_sum) AS FP_ZZSZYFP_SYQYGS_sum
           FROM (SELECT
                    XF_NSRSBH,
                    sum(JE) AS FP_ZZSZYFP_SYQYGS_sum
                  FROM ABC_TEMP
                  GROUP BY XF_NSRSBH) a, DJ_NSRXX b
           WHERE a.XF_NSRSBH = b.NSRSBH GROUP BY NSRDZDAH)
          p RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) q ON p.NSRDZDAH = q.nsrdzdah
        ''',
    'XFQC_NUM':  # 销方去重个数
        '''
        SELECT
          q.nsrdzdah,
          p.FP_ZZSZYFP_SYQYGS_sum as XFQC_NUM
        FROM
          (SELECT
             b.NSRDZDAH,
             sum(a.FP_ZZSZYFP_SYQYGS_sum) AS FP_ZZSZYFP_SYQYGS_sum
           FROM (SELECT
                    XF_NSRSBH,
                    count(DISTINCT (GF_NSRSBH)) AS FP_ZZSZYFP_SYQYGS_sum
                  FROM ABC_TEMP
                  GROUP BY XF_NSRSBH
                ) a, DJ_NSRXX b
           WHERE a.XF_NSRSBH = b.NSRSBH GROUP BY NSRDZDAH)
          p RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) q ON p.NSRDZDAH = q.nsrdzdah
        ''',
    'NGXZE':  # 年购项总额
        '''
        SELECT
          q.nsrdzdah,
          p.FP_ZZSZYFP_SYQYGS_sum as NGXZE
        FROM
          (SELECT
             b.NSRDZDAH,
             sum(a.FP_ZZSZYFP_SYQYGS_sum) AS FP_ZZSZYFP_SYQYGS_sum
           FROM (SELECT
                    GF_NSRSBH,
                    sum(JE) AS FP_ZZSZYFP_SYQYGS_sum
                  FROM ABC_TEMP
                  GROUP BY GF_NSRSBH        ) a, DJ_NSRXX b
           WHERE a.GF_NSRSBH = b.NSRSBH GROUP BY NSRDZDAH)
          p RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) q ON p.NSRDZDAH = q.nsrdzdah
        ''',
    'GFQC_NUM':  # 购方去重数量
        '''
        SELECT
          q.nsrdzdah,
          p.FP_ZZSZYFP_SYQYGS_sum as GFQC_NUM
        FROM
          (SELECT
             b.NSRDZDAH,
             sum(a.FP_ZZSZYFP_SYQYGS_sum) AS FP_ZZSZYFP_SYQYGS_sum
           FROM (SELECT
                    GF_NSRSBH,
                    count(DISTINCT (XF_NSRSBH)) AS FP_ZZSZYFP_SYQYGS_sum
                  FROM ABC_TEMP
                  GROUP BY GF_NSRSBH) a, DJ_NSRXX b
           WHERE a.GF_NSRSBH = b.NSRSBH GROUP BY NSRDZDAH)
          p RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) q ON p.NSRDZDAH = q.nsrdzdah
        ''',
    'SF_RATIO_AVG':  # 年度月份  税负平均变动率
        '''
        SELECT q.nsrdzdah,
        p.SB_ZZS_2003_ynse_ratio_avg AS SF_RATIO_AVG
        FROM
          (SELECT NSRDZDAH,
            AVG(ratio_ynse) AS SB_ZZS_2003_ynse_ratio_avg
          FROM (
            (SELECT a.NSRDZDAH,
              (b.sum_ynse/b.sum_yssr - a.sum_ynse/a.sum_yssr) / (a.sum_ynse/a.sum_yssr) AS ratio_ynse
            FROM
              (SELECT NSRDZDAH,
                SUM(ynse)                                AS sum_ynse,
                SUM(SYSL_XSE)                            AS sum_yssr,
                TO_CHAR(add_months(SSSQ_Q, 1), 'YYYYMM') AS rq
              FROM SB_ZZS_2003_YBNSR
              GROUP BY NSRDZDAH,
                TO_CHAR(add_months(SSSQ_Q, 1), 'YYYYMM')
              ) a,
              (SELECT NSRDZDAH,
                SUM(ynse)                 AS sum_ynse,
                SUM(SYSL_XSE)             AS sum_yssr,
                TO_CHAR(SSSQ_Q, 'YYYYMM') AS rq
              FROM SB_ZZS_2003_YBNSR
              GROUP BY NSRDZDAH,
                TO_CHAR(SSSQ_Q, 'YYYYMM')
              ) b
            WHERE a.NSRDZDAH = b.NSRDZDAH
            AND a.rq         = b.rq
            AND a.sum_ynse   > 0
            AND b.sum_ynse   > 0
            AND a.sum_yssr   > 0
            AND b.sum_yssr   >0
            )
          UNION ALL
            (SELECT a.NSRDZDAH,
              (b.sum_ynse/b.sum_yssr - a.sum_ynse/a.sum_yssr) / (a.sum_ynse/a.sum_yssr) AS ratio_ynse
            FROM
              (SELECT NSRDZDAH,
                SUM(ynse)                                AS sum_ynse,
                SUM(HWLW_XSE+GDZC_XSE)                   AS sum_yssr,
                TO_CHAR(add_months(SSSQ_Q, 1), 'YYYYMM') AS rq
              FROM SB_ZZS_XGMNSR_2005
              GROUP BY NSRDZDAH,
                TO_CHAR(add_months(SSSQ_Q, 1), 'YYYYMM')
              ) a,
              (SELECT NSRDZDAH,
                SUM(ynse)                 AS sum_ynse,
                SUM(HWLW_XSE+GDZC_XSE)    AS sum_yssr,
                TO_CHAR(SSSQ_Q, 'YYYYMM') AS rq
              FROM SB_ZZS_XGMNSR_2005
              GROUP BY NSRDZDAH,
                TO_CHAR(SSSQ_Q, 'YYYYMM')
              ) b
            WHERE a.NSRDZDAH = b.NSRDZDAH
            AND a.rq         = b.rq
            AND a.sum_ynse   > 0
            AND b.sum_ynse   > 0
            AND a.sum_yssr   > 0
            AND b.sum_yssr   >0
            ))
          GROUP BY NSRDZDAH
          ) p
        RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) q
        ON p.NSRDZDAH = q.nsrdzdah
        ''',
    'SF_AVG':  # 平均税负率
        '''
        SELECT
          q.nsrdzdah,
          p.SB_ZZS_SF_MEAN as SF_AVG
        FROM (SELECT
               NSRDZDAH,
               avg(sf) as SB_ZZS_SF_MEAN
             FROM
               ((SELECT
                  NSRDZDAH,
                  TO_CHAR(SSSQ_Q, 'YYYYMM') AS rq,
                  sum(ynse)/sum(SYSL_XSE) AS sf
                FROM SB_ZZS_2003_YBNSR
                GROUP BY NSRDZDAH,TO_CHAR(SSSQ_Q, 'YYYYMM')
                having sum(ynse)>0 and sum(SYSL_XSE)>0
                )union all(
                SELECT
                  NSRDZDAH,
                  TO_CHAR(SSSQ_Q, 'YYYYMM') AS rq,
                  sum(ynse)/sum(HWLW_XSE+GDZC_XSE) AS sf
                FROM SB_ZZS_XGMNSR_2005
                GROUP BY NSRDZDAH,TO_CHAR(SSSQ_Q, 'YYYYMM')
                having sum(ynse)>0 and sum(HWLW_XSE+GDZC_XSE)>0
                ))
                GROUP BY NSRDZDAH)
             p RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) q ON p.NSRDZDAH = q.nsrdzdah
        ''',
    'SJDKSE_RATIO_AVG':  # 年度月份实际 抵扣税额平均变动率
        '''
        SELECT
          q.nsrdzdah,
          p.SB_ZZS_2003_SJDKSE_ratio_avg as SJDKSE_RATIO_AVG
        FROM 
          (SELECT
             NSRDZDAH,
             avg(ratio_sjdkse) AS SB_ZZS_2003_SJDKSE_ratio_avg
           FROM
             (SELECT
                a.NSRDZDAH,
                (b.sum_sjdkse - a.sum_sjdkse) / a.sum_sjdkse AS ratio_sjdkse
              FROM
                (SELECT
                   NSRDZDAH,
                   sum(SJDKSE)                              AS sum_sjdkse,
                   TO_CHAR(add_months(SSSQ_Q, 1), 'YYYYMM') AS rq
                 FROM SB_ZZS_2003_YBNSR
                 GROUP BY NSRDZDAH, TO_CHAR(add_months(SSSQ_Q, 1), 'YYYYMM')) a,
                (SELECT
                   NSRDZDAH,
                   sum(SJDKSE)               AS sum_sjdkse,
                   TO_CHAR(SSSQ_Q, 'YYYYMM') AS rq
                 FROM SB_ZZS_2003_YBNSR
                 GROUP BY NSRDZDAH, TO_CHAR(SSSQ_Q, 'YYYYMM')) b
              WHERE a.NSRDZDAH = b.NSRDZDAH AND a.rq = b.rq AND a.sum_sjdkse > 0 AND b.sum_sjdkse > 0)
           GROUP BY NSRDZDAH)
          p RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) q ON p.NSRDZDAH = q.nsrdzdah
        ''',
    # 税负率=当期应纳增值税/当期应税销售收入
    # 当期应纳增值税=当期销项税额-实际抵扣进项税额
    # 实际抵扣进项税额=期初留抵进项税额+本期进项税额-进项转出-出口退税-期末留抵进项税额
    'YNSE_RATIO_AVG':  # 年度月份应纳税额平均变动率
        '''
        SELECT q.nsrdzdah,
          p.SB_ZZS_2003_ynse_ratio_avg AS YNSE_RATIO_AVG
        FROM
          (SELECT NSRDZDAH,
            AVG(ratio_ynse) AS SB_ZZS_2003_ynse_ratio_avg
          FROM (
            (SELECT a.NSRDZDAH,
              (b.sum_ynse - a.sum_ynse) / a.sum_ynse AS ratio_ynse
            FROM 
              (SELECT NSRDZDAH,
                SUM(ynse)                                AS sum_ynse,
                TO_CHAR(add_months(SSSQ_Q, 1), 'YYYYMM') AS rq
              FROM SB_ZZS_2003_YBNSR
              GROUP BY NSRDZDAH,
                TO_CHAR(add_months(SSSQ_Q, 1), 'YYYYMM')
              ) a,
              (SELECT NSRDZDAH,
                SUM(ynse)                 AS sum_ynse,
                TO_CHAR(SSSQ_Q, 'YYYYMM') AS rq
              FROM SB_ZZS_2003_YBNSR
              GROUP BY NSRDZDAH,
                TO_CHAR(SSSQ_Q, 'YYYYMM')
              ) b
            WHERE a.NSRDZDAH = b.NSRDZDAH
            AND a.rq         = b.rq
            AND a.sum_ynse   > 0
            AND b.sum_ynse   > 0)
            UNION ALL
              (SELECT a.NSRDZDAH,
                (b.sum_ynse - a.sum_ynse) / a.sum_ynse AS ratio_ynse
              FROM
                (SELECT NSRDZDAH,
                  SUM(ynse)                                AS sum_ynse,
                  TO_CHAR(add_months(SSSQ_Q, 1), 'YYYYMM') AS rq
                FROM SB_ZZS_XGMNSR_2005
                GROUP BY NSRDZDAH,
                  TO_CHAR(add_months(SSSQ_Q, 1), 'YYYYMM')
                ) a,
                (SELECT NSRDZDAH,
                  SUM(ynse)                 AS sum_ynse,
                  TO_CHAR(SSSQ_Q, 'YYYYMM') AS rq
                FROM SB_ZZS_XGMNSR_2005
                GROUP BY NSRDZDAH,
                  TO_CHAR(SSSQ_Q, 'YYYYMM')
                ) b
              WHERE a.NSRDZDAH = b.NSRDZDAH
              AND a.rq         = b.rq
              AND a.sum_ynse   > 0
              AND b.sum_ynse   > 0
              )
            )
          GROUP BY NSRDZDAH) p
        RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) q
        ON p.NSRDZDAH = q.nsrdzdah
        ''',
    'XFJSHJ_MEAN':  # 年度月份销方价税合计 平均
        '''
        SELECT
          q.nsrdzdah,
          p.FP_ZZSZYFP_XFJSHJ_mean as XFJSHJ_MEAN
        FROM
          (SELECT
             NSRDZDAH,
             avg(sum_jshj) AS FP_ZZSZYFP_XFJSHJ_mean
           FROM (SELECT
                   XF_NSRSBH,
                   SUM(JSHJ) AS sum_jshj
                 FROM ABC_TEMP
                 GROUP BY XF_NSRSBH, TO_CHAR(KPRQ, 'YYYYMM')) a, DJ_NSRXX b
           WHERE a.XF_NSRSBH = b.NSRSBH
           GROUP BY b.NSRDZDAH)
          p RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) q ON p.NSRDZDAH = q.nsrdzdah
        ''',
    'XFJSHJ_MEDIAN':  # 年度月份销方价税合计 中位数
        '''
        SELECT
          q.nsrdzdah,
          p.FP_ZZSZYFP_XFJSHJ_median as XFJSHJ_MEDIAN
        FROM
          (SELECT
             NSRDZDAH,
             MEDIAN(sum_jshj) AS FP_ZZSZYFP_XFJSHJ_median
           FROM (SELECT
                   XF_NSRSBH,
                   SUM(JSHJ) AS sum_jshj
                 FROM ABC_TEMP
                 GROUP BY XF_NSRSBH, TO_CHAR(KPRQ, 'YYYYMM')) a, DJ_NSRXX b
           WHERE a.XF_NSRSBH = b.NSRSBH
           GROUP BY b.NSRDZDAH)
          p RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) q ON p.NSRDZDAH = q.nsrdzdah
        ''',
    'IS_JC':  # 是否被稽查
        '''
        select b.nsrdzdah,(case when a.WTBZ = 'Y' then 1 else 0 end) as IS_JC from
        (SELECT
                JC.NSRDZDAH,
                MAX (JC.WTBZ) AS WTBZ
            FROM
                JC_AJXX JC
            WHERE
                EXTRACT (YEAR FROM jc.LRRQ) < 2015
            GROUP BY
                NSRDZDAH) a  RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) b ON a.NSRDZDAH = b.nsrdzdah
        ''',
    'AJLY':  # 案件来源类型
        '''
       SELECT b.nsrdzdah,
          a.AJLY
        FROM
          (SELECT JC.NSRDZDAH,
            MAX (JC.AJLY_DM) AS AJLY
          FROM JC_AJXX JC
          WHERE EXTRACT (YEAR FROM jc.LRRQ) < 2016
          GROUP BY NSRDZDAH
          ) a
        RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) b
        ON a.NSRDZDAH = b.nsrdzdah       
       ''',

    'XSFYL':  # 销售费用率
        '''
        SELECT
          q.nsrdzdah,
          p.SSFYLL as SSFYL
        FROM
        (select NSRDZDAH,avg(SSFYL) as SSFYLL from
          (SELECT
              NSRDZDAH,
              (XSFY+GLFY+CWFY)/YYCB as SSFYL
             FROM SB_NDSDS_2014 a 
             where EXTRACT (YEAR FROM sssq_q) <2016 and EXTRACT(YEAR FROM sssq_z) >2014
             and YYCB>0 
             )
             group by NSRDZDAH
             )
          p RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) q ON p.NSRDZDAH = q.nsrdzdah
        ''',
    'LRL':  # 利润率
        '''
        SELECT
          q.nsrdzdah,
          p.LRLL as LRL
        FROM
        (select NSRDZDAH,avg(LRL) as LRLL from
          (SELECT 
              NSRDZDAH,
              LRZE/YYCB as LRL
             FROM SB_NDSDS_2014 a 
             where EXTRACT (YEAR FROM sssq_q) <2016 and EXTRACT(YEAR FROM sssq_z) >2014
             and YYCB>0 
             )
             group by NSRDZDAH
             )
          p RIGHT JOIN (SELECT NSRDZDAH FROM lg_data14_15_BIG WHERE YEARS=2015) q ON p.NSRDZDAH = q.nsrdzdah
        ''',
    'WTBZ':  # 企业标签
        '''
        SELECT nsrdzdah,(case when WTBZ = 'Y' then 1 else 0 end) as WTBZ from lg_data14_15_BIG WHERE YEARS=2015
        '''
}
