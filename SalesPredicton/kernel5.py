import myutils
from itertools import product   
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import numpy as np # linear algebra
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import date
import time
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pickle
import h5py
import sys
use_tree= True
use_linear = False
use_linear_linear = False
use_prev_month = False
use_linear_meta = False

global now 
now= time.time()
def rt(prefix):
    global now
    print(prefix + " : " + str(time.time()-now) )
    now = time.time()

last_block = 33
lag = 12
w5 = pd.read_pickle('y6.pkl')


test_df = pd.read_csv('../input/test.csv')
train_columns = list(set(w5.columns)-set(['shop_name', 'item_name', 'item_category_name','item_cnt_month1']))
idx = (w5['date_block_num']<last_block) & (w5['date_block_num']>(last_block-lag))
X_train = w5[train_columns].loc[idx]
Y_train = w5['item_cnt_month1'].loc[idx  ]
X_test = w5[train_columns].loc[w5['date_block_num']==last_block]


print("Starting training and prediction" )
startTime = time.time()
if use_tree:
    # model = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
    #     gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10,
    #     min_child_weight=1, missing=None, n_estimators=10, nthread=-1,
    #     objective='multi:softmax', reg_alpha=0, reg_lambda=1,
    #     scale_pos_weight=1, seed=0, silent=True, subsample=.8, eval_metric ='rmse')
    # learninreate=0.2565963232397263, gamma=2.025792317074175, max_depth=37, n_estimator=226, min_child_weight=9.46056590169735, col_sample_by_tree=0.2006610890423738, sub_sample0.6724546065784309
    model = XGBRegressor(max_depth=37, min_child_weight = 9.46, n_estimators=250, scale_pos_weight=1, 
        learning_rate = 0.256, subsample=0.67, gamma=2)  
    model.fit(X_train, Y_train)    
    predictions = model.predict(X_test)
    
    print(dict(zip(X_train.columns,model.feature_importances_)))

if use_linear:
    print("Using linear")
    polynomial_features= PolynomialFeatures(degree=3)
    X_train_poly = polynomial_features.fit_transform(X_train)
    X_test_poly = polynomial_features.fit_transform(X_test)
    model = LinearRegression()
    model.fit(X_train_poly, Y_train)    
    predictions = model.predict(X_test_poly)
    
if use_linear_linear:
    print("Using linear")        
    model = LinearRegression()
    model.fit(X_train, Y_train)    
    predictions = model.predict(X_test)

if use_linear_meta:
    idx = (w5['date_block_num']<last_block) & (w5['date_block_num']>(last_block-lag))
    X_train_m = w5[train_columns].loc[idx]
    Y_train_m = w5['item_cnt_month1'].loc[idx]
    X_test_m = w5[train_columns].loc[w5['date_block_num']==last_block]
    Y_test_m = w5['item_cnt_month1'].loc[w5['date_block_num']==last_block ]

     # Creating two models and predict them
    ## 1  Linear
    model_1 = LinearRegression()
    model_1.fit(X_train_m, Y_train_m)    
    predictions_1 = model_1.predict(X_test_m)
    ## 2 Tree
    model_2 = XGBRegressor()        
    model_2.fit(X_train_m, Y_train_m)    
    predictions_2 = model_2.predict(X_test_m)

    # Create and train meta model    
    meta_train_df =  pd.DataFrame()
    meta_train_df['1'] = predictions_1
    meta_train_df['2'] = predictions_2
    model_meta = LinearRegression()    
    model_meta.fit(meta_train_df, Y_test_m)    
    
    # Applying the trained meta model on
    meta_test_df =  pd.DataFrame()
    meta_test_df['1'] =  model_1.predict(X_test)    
    meta_test_df['2'] =  model_2.predict(X_test)    
    predictions = model_meta.predict(meta_test_df)

    

if use_prev_month:
    predictions = X_test['item_cnt_month']*1 + X_test['item_cnt_prev_month']*0
print("Ending training" + str(time.time()-startTime))

# Create the submission file
X_test['pred'] = predictions
answer = pd.merge(test_df,X_test,on=['shop_id','item_id'],how='left').fillna(0)[['ID','pred']]
answer.columns =  ['ID','item_cnt_month']
answer['item_cnt_month']=answer['item_cnt_month'].clip(0,20)
answer.to_csv('csv_to_submit.csv', index = False)
