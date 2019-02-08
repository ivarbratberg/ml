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


www = pd.read_csv('y3')

# rt("www")
# # prev month column
# y_train = www[['date_block_num','item_cnt_month', 'shop_id','item_id']].copy()
# y_train['date_block_num'] += 1
# w4 = pd.merge(www, y_train, on=['shop_id','item_id','date_block_num'], how='left').fillna(0)
# w4 = w4.rename(columns={'item_cnt_month_x' : 'item_cnt_month', 'item_cnt_month_y' : 'item_cnt_next_month'})

# w4.to_csv('y4', index = False)
rt("w4")
# prev month column
prev_month = www[['date_block_num','item_cnt_month', 'shop_id','item_id']].copy()
prev_month['date_block_num'] -= 1
w5 = pd.merge(www,prev_month, on=['shop_id','item_id','date_block_num'], how='left').fillna(0)
w5 = w5.rename(columns={'item_cnt_month_x' : 'item_cnt_month', 'item_cnt_month_y' : 'item_cnt_prev_month'})
w5.to_csv('y5.csv', index = False)
