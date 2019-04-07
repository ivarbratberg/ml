from numpy import loadtxt
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
use_prev_month = False
use_linear_meta = True


global now 
now= time.time()
def rt(prefix):
    global now
    print(prefix + " : " + str(time.time()-now) )
    now = time.time()

last_block = 26
lag = 12
w6 = pd.read_pickle("./y6.pkl")
print(w6.columns)
# Take a subset
w6 = w6[(w6['shop_id'] == 0) & (w6['item_cnt_month'] > 0)]
# w6.to_csv('y7.csv', index = False)
# sys.exit()

train_columns = list(set(w6.columns)-set(['shop_name', 'item_name', 'item_category_name','item_cnt_month1']))

for block_nr in range(25,last_block):
    idx = (w6['date_block_num']<block_nr) & (w6['date_block_num']>(block_nr-lag))
    X_train = w6[train_columns].loc[idx]
    Y_train = w6['item_cnt_month1'].loc[idx  ]
    X_test = w6[train_columns].loc[w6['date_block_num']==block_nr]
    Y_test = w6['item_cnt_month1'].loc[w6['date_block_num']==block_nr]
    #Y_test = ww['item_cnt_day','item_id'].loc[ww['date_block_num']==block_nr ]


    #X_test.to_csv('xtest.csv')
    #Y_test.to_csv('ytest.csv')
    #X_train.to_csv('xtrain.csv')
    #Y_train.to_csv('ytrain.csv')
    #XGBoost
    print("Starting training" )
    startTime = time.time()
    if use_tree:        
        eval_set = [(X_train, Y_train), (X_test, Y_test)]
        eval_metric = ["rmse"]
        #time model.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)

        model = XGBRegressor(max_depth=4, min_child_weight = 10, n_estimators=100, scale_pos_weight=1, 
        learning_rate = 0.1, subsample=0.8, reg_alpha=0.3, gamma=100)        
        model.fit(X_train, Y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)    
        predictions = model.predict(X_test)
    
    if use_linear:
        polynomial_features= PolynomialFeatures(degree=2)
        X_train_poly = polynomial_features.fit_transform(X_train)
        X_test_poly = polynomial_features.fit_transform(X_test)
        model = LinearRegression()
        model.fit(X_train_poly, Y_train)    
        predictions = model.predict(X_test_poly)

    if use_linear_meta:
        idx = (w6['date_block_num']<(block_nr-1)) & (w6['date_block_num']>(block_nr-lag))
        X_train_m = w6[train_columns].loc[idx]
        Y_train_m = w6['item_cnt_month1'].loc[idx]
        X_test_m = w6[train_columns].loc[w6['date_block_num']==(block_nr-1)]
        Y_test_m = w6['item_cnt_month1'].loc[w6['date_block_num']==(block_nr-1) ]

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
        predictions = (X_test['item_cnt_month']*0.7 + X_test['item_cnt_prev_month'])*0.3/2
    print("Ending training" + str(time.time()-startTime))

    # add back the true value and the predicted value to make it easy to aggregate
    X_test['y'] = Y_test
    X_test['pred'] = predictions
    X_test['current'] = w6['item_cnt_month'].loc[w6['date_block_num']== block_nr]
    # X_test.to_csv('xtest.csv', index = False)
    # 
    # we add
    #pred_month = X_test.groupby(['date_block_num','shop_id','item_id'], as_index=False).agg({'y' : 'sum', 'item_cnt_day' : 'sum'})
    #pred_month.to_csv('pred_month.csv')
    
    #answer = pd.merge(Y_test, predictions, on=['shop_id','item_id'],how='left')
    score=np.sqrt(np.sum((X_test['y'].clip(0,20)-X_test['pred'].clip(0,20))**2)/predictions.size)
    score2=np.sqrt(np.sum((X_test['y'].clip(0,20)-X_test['current'].clip(0,20))**2)/predictions.size)
    print("Score " + str(block_nr) + " "+str(score) + " " + str(score2))

