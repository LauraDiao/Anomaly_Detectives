import numpy as np
import json
import sys
import pandas as pd
import os
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingRegressor
import warnings
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
warnings.filterwarnings("ignore")

from helper import *

def test_feat(cond, df, cols, p, df_u): 
    unseen = ''
    if cond =='unseen': 
        unseen = 'unseen'
    # col is feauture comb
    # p is for loss or latency   1: loss  # 2 : latency
    X = df[cols]
    
    X2 = df_u[cols]

    if p == 1:  # flag found in test_mse
        y = df.loss
        y2 = df_u.loss
    if p == 2: 
        y = df.latency
        y2 = df_u.latency
        
    # randomly split into train and test sets, test set is 80% of data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    if unseen == 'unseen': 
        X_test = X2
        y_test = y2
    
    clf = DecisionTreeRegressor()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    #acc1 = mean_squared_error(y_test, y_pred)
    acc1 = clf.score(X_test, y_test)
    
    clf2 = RandomForestRegressor(n_estimators=200, n_jobs = -1)
    clf2 = clf2.fit(X_train,y_train)
    y_pred2 = clf2.predict(X_test)
    #acc2= mean_squared_error(y_test, y_pred2)
    acc2 = clf2.score(X_test, y_test)
    
    clf3 = ExtraTreesRegressor(n_estimators=200, n_jobs = -1)
    clf3 = clf3.fit(X_train,y_train)
    y_pred3 = clf3.predict(X_test)
    #acc3= mean_squared_error(y_test, y_pred3)
    acc3 = clf3.score(X_test, y_test)
    
#     pca = PCA(n_components = 2) 
#     X_transformed = pca.fit_transform(X_train) 
#     clf4 = ExtraTreesRegressor(n_estimators=100, n_jobs = -1)
#     clf4 = clf4.fit(X_transformed, y_train)
#     newdata_transformed = pca.transform(X_test)
#     y_pred4 = clf4.predict(newdata_transformed)
#     #acc4 = mean_squared_error(y_test, y_pred4)
#     acc4 = clf4.score(X_test, y_test)
    
    clf_gbc = GradientBoostingRegressor(random_state=0, max_depth = 6, n_estimators=200)
    clf_gbc.fit(X_train, y_train)
    y_pred5 = clf_gbc.predict(X_test)
    #acc5 = mean_squared_error(y_test, y_pred5) 
    acc5 = clf_gbc.score(X_test, y_test)
    return [acc1, acc2, acc3,  acc5]


def test_mse(cond, all_comb1, all_comb2):
    unseen = ''
    if cond =='unseen': 
        unseen = 'unseen'
    filedir_unseen = os.path.join(os.getcwd(), "outputs", unseen + "combined_transform.csv")
    df_unseen = pd.read_csv(filedir_unseen)
    filedir = os.path.join(os.getcwd(), "outputs", "combined_transform.csv")
    df = pd.read_csv(filedir)
    
    all_comb1 = pd.Series(all_comb1).apply(lambda x: list(x))
    all_comb2 = pd.Series(all_comb2).apply(lambda x: list(x))
    
    dt = []
    rf = []
    et = []
    #pca = []
    gbc = []
    for i in all_comb1:
        acc_loss = test_feat(cond, df, i, 1, df_unseen)
        dt.append(acc_loss[0])  
        rf.append(acc_loss[1])  
        et.append(acc_loss[2])   
        #pca.append(acc_loss[3])   
        gbc.append(acc_loss[3])
        
    # optimze by adding a flag called losslat to avoid making two dataframes of results
    dt2 = []
    rf2 = []
    et2 = []
    #pca2 = []
    gbc2 = []
    for i in all_comb2:
        # 1 = loss
        # 2 = latency
        acc_latency = test_feat(cond, df, i, 2, df_unseen)
        dt2.append(acc_latency[0])
        rf2.append(acc_latency[1])
        et2.append(acc_latency[2]) 
        #pca2.append(acc_latency[3])
        gbc2.append(acc_latency[3])
        
    dict1 = pd.DataFrame({'feat': all_comb1, 'dt': dt, 'rf': rf, 'et': et, 'gbc': gbc})
    dict2 = pd.DataFrame({'feat2': all_comb2, 'dt2': dt2, 'rf2': rf2, 'et2': et2, 'gbc2': gbc2})
    
    path = os.path.join(os.getcwd() , "outputs")
    dict1.to_csv(os.path.join(path, unseen + "feat_df1.csv"), index = False)
    dict2.to_csv(os.path.join(path, unseen + "feat_df2.csv"), index = False)


def best_performance(cond):
    unseen = ''
    if cond == 'unseen': 
        unseen = 'unseen'
    #print("finding best loss performance")
    filedir1 = os.path.join(os.getcwd(), "outputs", unseen + "feat_df1.csv")
    df1 = pd.read_csv(filedir1)
    df1_round = df1.round(decimals = 3)
    print( "\n")
    #print("Loss Performance sorted from highest to lowest metric: r2", "\n")
    print("Best performance for Loss Models")
    dt_p1 = df1_round.sort_values(by=['dt'], ascending = False)
    print(dt_p1[:2], '\n' )
    dt_p2 = df1_round.sort_values(by=['rf'], ascending = False)
    print(dt_p2[:2], '\n' )
    dt_p3 = df1_round.sort_values(by=['et'], ascending = False)
    print(dt_p3[:2], '\n' )
    dt_p4 = df1_round.sort_values(by=['gbc'], ascending = False)
    print(dt_p4[:2], '\n' )
    
    #print("finding best latency performance")
    filedir2 = os.path.join(os.getcwd(), "outputs", unseen + "feat_df2.csv")
    df2 = pd.read_csv(filedir2)
    df2_round = df2.round(decimals = 3)
    #print("Latency Performance sorted from highest to lowest metric: r2", "\n")
    #print(df2_round.sort_values(by=['dt2', 'rf2', 'et2', 'gbc2'], ascending = False)[:5], "\n")
    print("Best performance for Latency Models")
    dt2_p1 = df2_round.sort_values(by=['dt2'], ascending = False)
    print(dt2_p1[:2], '\n' )
    dt2_p2 = df2_round.sort_values(by=['rf2'], ascending = False)
    print(dt2_p2[:2], '\n' )
    dt2_p3 = df2_round.sort_values(by=['et2'], ascending = False)
    print(dt2_p3[:2], '\n' )
    dt2_p4 = df2_round.sort_values(by=['gbc2'], ascending = False)
    print(dt2_p4[:2], '\n' )



def getAllCombinations( cond_):
    lst =  ['total_bytes','max_bytes', "1->2Bytes",'2->1Bytes'
                ,'1->2Pkts','2->1Pkts','total_pkts','number_ms', 'pkt_ratio','time_spread', 'pkt sum','longest_seq'
                ,'total_pkt_sizes', 'mean_tdelta', 'max_tdelta']# 'proto',
    latency_lst = ['byte_ratio', 'pkt_ratio', 'time_spread', 'total_bytes', '2->1Pkts']
    loss_lst = ['total_pkts', 'total_pkt_sizes', '2->1Bytes', 'number_ms', 'mean_tdelta', 'max_tdelta'] 
  
    if cond_ == 1:
        lst = loss_lst
    if cond_ == 2:
        lst = latency_lst
    uniq_objs = set(lst)
    combinations = []
    for obj in uniq_objs:
        for i in range(0,len(combinations)):
            combinations.append(combinations[i].union([obj]))
        combinations.append(set([obj]))
    print("all combinations generated")
    return combinations

def feat_impt(labl):
    label_col = labl
    
    df = pd.read_csv(os.path.join(os.getcwd() , "outputs", 'combined_transform.csv'))

    indexcol = ['total_bytes','max_bytes','2->1Bytes','2->1Pkts','total_pkts', 'total_pkts_min', 'total_pkts_max', 'number_ms', 'pkt_ratio','time_spread', 'time_spread_min','time_spread_max','pkt sum','longest_seq', 'longest_seq_min', 'longest_seq_max','total_pkt_sizes','byte_ratio', 'mean_tdelta', 'max_tdelta']
    len(indexcol)

    X_train, X_test, y_train, y_test = train_test_split(
        df[[x for x in indexcol if x in df.columns]], df[label_col])
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    etree = ExtraTreesRegressor(n_estimators=400, n_jobs=4)
    etreeft = etree.fit(X_train,y_train)

    y_pred3 = etree.predict(X_test)
    acc3= mean_squared_error(y_test, y_pred3)

    print(f'mse: {acc3}, r2: {etree.score(X_test, y_test)}')
    feat_imp = pd.Series(index=[x for x in indexcol if x in df.columns], 
              data=etree.feature_importances_).sort_values(ascending=False)
    return feat_imp
    
def rolling_window(labl):
    # rolling wdinow
#     Pd rolling window
#     Aggregate last 10 rows
#     Other idea: shift row and calc percent difference

    label_col = labl
    
    df = pd.read_csv(os.path.join(os.getcwd() , "outputs", 'combined_transform.csv'))

    indexcol = ['total_bytes','max_bytes','2->1Bytes','2->1Pkts','total_pkts', 'total_pkts_min', 'total_pkts_max', 'number_ms', 'pkt_ratio','time_spread', 'time_spread_min','time_spread_max','pkt sum','longest_seq', 'longest_seq_min', 'longest_seq_max','total_pkt_sizes','byte_ratio', 'mean_tdelta', 'max_tdelta']
    len(indexcol)

    X_train, X_test, y_train, y_test = train_test_split(
        df[[x for x in indexcol if x in df.columns]], df[label_col])
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    etree = ExtraTreesRegressor(n_estimators=400, n_jobs=4)
    etreeft = etree.fit(X_train,y_train)

    y_pred3 = etree.predict(X_test)
    acc3= mean_squared_error(y_test, y_pred3)
    roll = pd.Series(y_pred3).rolling(2).mean() # roling over past 20 seconds
    # print(roll)

    print(f'mse: {acc3}, r2: {etree.score(X_test, y_test)}')
    feat_imp = pd.Series(index=[x for x in indexcol if x in df.columns], 
              data=etree.feature_importances_).sort_values(ascending=False)
    
    return feat_imp