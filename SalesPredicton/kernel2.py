import myutils
from numpy import loadtxt
from xgboost import XGBClassifier
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
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pickle
import h5py
import sys
import gc

global now 
now= time.time()
def rt(prefix):
    global now
    print(prefix + " : " + str(time.time()-now) )
    now = time.time()

    

www = pd.read_csv('y3')

# Save memory
www['date_block_num'] = www['date_block_num'].astype(np.int8)
www['shop_id'] = www['shop_id'].astype(np.int8)
www['item_id'] = www['item_id'].astype(np.int16)
www['item_cnt_month'] = www['item_cnt_month'].astype(np.float16)
www['item_price'] = www['item_cnt_month'].astype(np.float16)
www['item_category_id'] = www['item_category_id'].astype(np.int8)

rt("www")
www['item_cnt_month']=www['item_cnt_month'].clip(0,20)



www = myutils.mean_encode(www,'item_cnt_month', ['date_block_num','item_category_id'], 'date_category_avg_cnt' )
www = myutils.create_lags2(www, 'date_block_num', ['item_category_id'], 'date_category_avg_cnt', (-6,-1))

www = myutils.mean_encode(www,'item_cnt_month', ['date_block_num','item_id'], 'date_item_avg_cnt' )
www = myutils.create_lags2(www, 'date_block_num', ['item_id'], 'date_item_avg_cnt', (-6,-1))

www = myutils.mean_encode(www,'item_cnt_month', ['date_block_num','item_category_id'], 'date_cat_avg_cnt' )
www = myutils.create_lags2(www, 'date_block_num', ['item_category_id'], 'date_cat_avg_cnt', (-6,-1))

www = myutils.mean_encode(www,'item_cnt_month', ['date_block_num','shop_id'], 'date_shop_avg_cnt' )
www = myutils.create_lags2(www, 'date_block_num', ['shop_id'], 'date_shop_avg_cnt', (-6,-1))

www = myutils.mean_encode(www,'item_cnt_month', ['date_block_num','shop_id', 'item_category_id'], 'date_shop_cat_avg_cnt' )
www = myutils.create_lags2(www, 'date_block_num', ['shop_id', 'item_category_id'], 'date_shop_cat_avg_cnt', (-6,-1))
# For this triplet we allready have the count
www = myutils.create_lags2(www, 'date_block_num', ['shop_id', 'item_id'], 'item_cnt_month', (-6,-1))



www['month'] = www['date_block_num'] % 12
# Get the mean price as a columns
mean_price = www.groupby(['item_id'], as_index=False).agg({ 'item_price': 'mean'})
mean_price = mean_price.rename(columns={'item_price' : 'item_mean_price'})
www = pd.merge(www,mean_price,on=['item_id'], how='left')
# find the price divided by mean
www['item_relative_price'] = www['item_price']/www['item_mean_price']

www.to_pickle("./y5.pkl")

rt("w5")
