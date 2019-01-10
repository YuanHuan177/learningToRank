# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  @Time : 2019/1/10 17:29
#  @Author : lg
#  @File : ComELearning.py
#  网络属性与基本指标属性结合 表征计算

import operator
import os
import time

import pandas as pd
import numpy as np
from itertools import chain
from scipy.optimize import minimize
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss, \
    precision_recall_curve, roc_curve, average_precision_score

from sklearn.externals import joblib

