from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import date
import time
from datetime import datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pickle
import h5py
import sys

global now 
now= time.time()
def rt(prefix):
    global now
    print(prefix + " : " + str(time.time()-now) )
    now = time.time()

def mean_encode(m,col,cols):
    grouped = m.groupby(cols, as_index=False).agg({ col: 'mean'})
    grouped = grouped.rename(columns={col : col+'_mean'})
    return pd.merge(m,grouped,on=cols, how='left').fillna(0)

def create_lags(m, key_cols, value_col, max_lag):
    key_cols.append('date_block_num')
    for lag in range (1, max_lag+1):
        prev_month = m[key_cols+[value_col]].copy()
        prev_month = prev_month.rename(columns={value_col : value_col + str(lag)})
        prev_month['date_block_num'] += lag
        m = pd.merge(m, prev_month, on=['shop_id','item_id','date_block_num'], how='left').fillna(0)

    


www = pd.read_csv('y3')
rt("www")
www['item_cnt_month']=www['item_cnt_month'].clip(0,20)


# Get the mean price as a columns
mean_price = www.groupby(['item_id'], as_index=False).agg({ 'item_price': 'mean'})
mean_price = mean_price.rename(columns={'item_price' : 'item_mean_price'})
www = pd.merge(www,mean_price,on=['item_id'], how='left')
www = www.rename(columns={'item_cnt_day' : 'item_cnt_month'})
# find the price divided by mean
www['item_relative_price'] = www['item_price']/www['item_mean_price']

# for month in www['date_block_num'].drop_duplicates():   
#     shop = w5[w5.date_block_num == month]['shop_id'].drop_duplicates()   
#     item = w5[w5.date_block_num == month]['item_id'].drop_duplicates()   
#     grid.append( np.asarray(   list( product( *[shop,item,[month]] ) )    )  )     


# -4 month column
prev_month = www[['date_block_num','item_cnt_month', 'shop_id','item_id']].copy()
prev_month = prev_month.rename(columns={'item_cnt_month' : 'item_cnt_prev4_month'})
prev_month['date_block_num'] += 4
www = pd.merge(www, prev_month, on=['shop_id','item_id','date_block_num'], how='left').fillna(0)
rt("mont -4")

# -3 month column
prev_month = www[['date_block_num','item_cnt_month', 'shop_id','item_id']].copy()
prev_month = prev_month.rename(columns={'item_cnt_month' : 'item_cnt_prev3_month'})
prev_month['date_block_num'] += 3
www = pd.merge(www, prev_month, on=['shop_id','item_id','date_block_num'], how='left').fillna(0)
rt("mont -3")


# -2 month column
prev_month = www[['date_block_num','item_cnt_month', 'shop_id','item_id']].copy()
prev_month = prev_month.rename(columns={'item_cnt_month' : 'item_cnt_prev2_month'})
prev_month['date_block_num'] += 2
www = pd.merge(www, prev_month, on=['shop_id','item_id','date_block_num'], how='left').fillna(0)
rt("mont -2")

# -1 month column
prev_month = www[['date_block_num','item_cnt_month', 'shop_id','item_id']].copy()
prev_month = prev_month.rename(columns={'item_cnt_month' : 'item_cnt_prev_month'})
prev_month['date_block_num'] += 1
w4 = pd.merge(www, prev_month, on=['shop_id','item_id','date_block_num'], how='left').fillna(0)



w4.to_csv('y5.csv', index = False)
rt("w5")
