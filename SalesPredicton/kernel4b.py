from numpy import loadtxt
from xgboost import XGBRegressor
import xgboost as xgb
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
use_tree= True
use_linear = False
use_prev_month = False
use_linear_meta = True
use_subset= False


global now 
now= time.time()
def rt(prefix):
    global now
    print(prefix + " : " + str(time.time()-now) )
    now = time.time()

last_block = 33
lag = 12
w6 = pd.read_pickle("./y6.pkl")

# Take a subset
if use_subset:
    w6 = w6[(w6['shop_id'] == 0) & (w6['item_cnt_month'] > 0)]
    #w6.to_csv('y7.csv', index = False)


train_columns = list(set(w6.columns)-set(['shop_name', 'item_name', 'item_category_name','item_cnt_month1']))

folds = []
data_dmatrix = xgb.DMatrix(data=w6[train_columns],label=w6['item_cnt_month1'])


for block_nr in range(25,last_block):
    idx_in = (w6['date_block_num']<block_nr) & (w6['date_block_num']>(block_nr-lag))
    idx_out = w6['date_block_num']==block_nr
    folds.append((idx_out, idx_in))
    
# params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
#                 'max_depth': 5, 'alpha': 10}

regressor =  xgb.XGBRegressor(max_depth=4, min_child_weight = 10, n_estimators=100, scale_pos_weight=1, 
        learning_rate = 0.1, subsample=0.8, reg_alpha=0.3, gamma=100)        

params = regressor.get_xgb_params()

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, folds=folds,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
    
print(cv_results)
print(datetime.now())
    
