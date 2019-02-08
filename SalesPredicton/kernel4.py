from itertools import product   
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import numpy as np # linear algebra
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
use_tree= False
use_linear = False
use_prev_month = False
use_linear_meta = True

global now 
now= time.time()
def rt(prefix):
    global now
    print(prefix + " : " + str(time.time()-now) )
    now = time.time()

last_block = 33
w5 = pd.read_csv('y5.csv')
grid = []    
lag = 10

# merge in zero items
for month in w5['date_block_num'].drop_duplicates():   
    shop = w5[w5.date_block_num == month]['shop_id'].drop_duplicates()   
    item = w5[w5.date_block_num == month]['item_id'].drop_duplicates()   
    grid.append( np.asarray(   list( product( *[shop,item,[month]] ) )    )  )     
cols = ['shop_id','item_id','date_block_num']   
grid = pd.DataFrame(np.vstack(grid), columns = cols, dtype=np.int32)    
w5 = pd.merge(grid,w5, on = cols, how = 'left').fillna(0)    

# merge in category id
item_categories_df = pd.read_csv('../input/item_categories.csv')
shops_df = pd.read_csv('../input/shops.csv')
items_df = pd.read_csv('../input/items.csv')
w5 = pd.merge(w5, items_df, on="item_id")
w5 = pd.merge(w5, item_categories_df, on="item_category_id")
w5 = pd.merge(w5, shops_df, on="shop_id")

test_df = pd.read_csv('../input/test.csv')
train_columns = list(set(w5.columns)-set(['shop_name', 'item_name', 'item_category_name', 'item_cnt_month']))
idx = (w5['date_block_num']<last_block) & (w5['date_block_num']>(last_block-lag))
X_train = w5[train_columns].loc[idx]
Y_train = w5['item_cnt_month'].loc[idx  ]
X_test = w5[train_columns].loc[w5['date_block_num']==last_block]


# Clip values before making predictions
X_train['item_cnt_prev_month']=X_train['item_cnt_prev_month'].clip(0,20)
X_train['item_cnt_prev2_month']=X_train['item_cnt_prev2_month'].clip(0,20)
X_train['item_cnt_prev3_month']=X_train['item_cnt_prev3_month'].clip(0,20)

print("Starting training and prediction" )
startTime = time.time()
if use_tree:
    # model = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
    #     gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10,
    #     min_child_weight=1, missing=None, n_estimators=10, nthread=-1,
    #     objective='multi:softmax', reg_alpha=0, reg_lambda=1,
    #     scale_pos_weight=1, seed=0, silent=True, subsample=.8, eval_metric ='rmse')
    model = XGBRegressor(max_depth=8, min_child_weight = 300, n_estimators=1000, scale_pos_weight=1, 
        learning_rate = 0.1, subsample=0.8, reg_alpha=0.3, gamma=100)  
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
    
if use_linear_meta:
    idx = (w5['date_block_num']<last_block) & (w5['date_block_num']>(last_block-lag))
    X_train_m = w5[train_columns].loc[idx]
    Y_train_m = w5['item_cnt_month'].loc[idx]
    X_test_m = w5[train_columns].loc[w5['date_block_num']==last_block]
    Y_test_m = w5['item_cnt_month'].loc[w5['date_block_num']==last_block ]

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