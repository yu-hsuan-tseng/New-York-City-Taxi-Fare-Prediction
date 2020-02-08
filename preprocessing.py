#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 21:05:15 2020
@author: yu_hsuantseng
@project : kaggle- new york taxi fare prediction

"""

import copy 
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time


"""
    Read in original file for data preprocessing and data cleaning
    Includes EDA
"""

def readin(file_path):
    data = pd.read_csv(file_path)
    data.index = np.arange(1,len(data)+1)
    return data



def preview(data):
    
    print("data shape: ",data.shape)
    print("columns: ",data.columns)
    print("statistic description:")
    print(data.describe())
    null_rate = data.isnull().sum() / len(data)
    print("null rate is:",null_rate)
    
    
"""
    主程式位置,為方便之後優化程式碼以及演算法
    將可能會使用到的函式以物件導向的方式進行設計
    
"""

def feature_1(data):
    
    feature_1 = {'date':[],'time':[]}
    for t in data['key']:
        t = t.split(" ")
        date = t[0]
        t = t[1].split(".")
        t = t[0]
        feature_1['date'].append(date)
        feature_1['time'].append(t[:2])
    data["date"] = feature_1['date']
    data['time'] = feature_1['time']
    
    return data
    


def feature_2(data):
    
    day_night = []
    
    for t in data['time']:
        t = int(t)
        if t > 6 and t < 23 :
            day_night.append(0)
        else:
            day_night.append(1)
    data['daytime_nightime'] = day_night
    return data



    
    
if __name__=="__main__":
    
    
    error_rate = 0
    '''
    try:
        file_path = "train.csv"
        df_train = readin(file_path)
        preview(df_train)
    except:
        print("function design error exception !")
        error_rate+=1
        pass
    print("function design error :",error_rate)
    df_train = df_train.dropna()
    
    df_train = feature_1(df_train)
    df_train = df_train.drop(['key','pickup_datetime'],axis=1)
    df_train = feature_2(df_train)
    df_train = df_train.drop(['daytime_nightime'],axis=1)
    df_train = feature_2(df_train)
    
    '''
    print("phase one completed")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    