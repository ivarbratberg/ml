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
import os
import pickle
import h5py
import sys
import pdb

global now 
now= time.time()
def rt(prefix):
    global now
    print(prefix + " : " + str(time.time()-now) )
    now = time.time()

w5 = pd.read_pickle('y5.pkl')
for col in w5.columns:
    if w5[col].dtype == np.float64:
        w5[col] = w5[col].astype(np.float32)

grid = []    
lag = 6


# Create lag -1 for item count which we should test againts
w5 = myutils.create_lags2(w5, 'date_block_num', ['shop_id', 'item_id'], 'item_cnt_month', (1,1))

# merge in zero items
for month in w5['date_block_num'].drop_duplicates():   
    shop = w5[w5.date_block_num == month]['shop_id'].drop_duplicates()   
    item = w5[w5.date_block_num == month]['item_id'].drop_duplicates()   
    grid.append( np.asarray(   list( product( *[shop,item,[month]] ) )    )  )     
cols = ['shop_id','item_id','date_block_num']   
grid = pd.DataFrame(np.vstack(grid), columns = cols, dtype=np.int32)    
w5 = pd.merge(grid,w5, on = cols, how = 'left').fillna(0)    
pdb.set_trace()
print("Starting storing the file")

print(w5.dtypes)
w5.to_pickle("./y6.pkl")
#w5.to_csv('y6head.csv', index = False)
