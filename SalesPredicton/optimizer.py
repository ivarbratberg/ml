import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import StratifiedKFold
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
import pdb
import os
import pickle
import h5py
import sys

# https://www.kaggle.com/omarito/gridsearchcv-xgbregressor-0-556-lb


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

w6 = w6[(w6['shop_id'] < 10) & (w6['item_cnt_month'] > 0)]

# w6.to_csv('y7.csv', index = False)
# sys.exit()
train_columns = list(set(w6.columns)-set(['shop_name', 'item_name', 'item_category_name','item_cnt_month1']))

block_nr = last_block
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

eval_set = [(X_train, Y_train), (X_test, Y_test)]
eval_metric = ["rmse"]
#time model.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)

#XGBRegressor(max_depth=8, min_child_weight = 300, n_estimators=1000, scale_pos_weight=1, 
#learning_rate = 0.1, subsample=0.8, reg_alpha=0.3, gamma=100) 

# A parameter grid for XGBoost
params = {'min_child_weight':[1,10,20,30,50,100,200,300,400], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,3,4,6,10]}
xgb = XGBRegressor(nthread=-1)

grid = GridSearchCV(xgb, params)
grid.fit(X_train, Y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)
predictions = grid.best_estimator_.predict(X_test)

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


model = XGBRegressor(max_depth=4, min_child_weight = 10, n_estimators=100, scale_pos_weight=1, 
learning_rate = 0.1, subsample=0.8, reg_alpha=0.3, gamma=100)        
model.fit(X_train, Y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)    
predictions = model.predict(X_test)


