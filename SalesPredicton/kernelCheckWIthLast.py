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


www = pd.read_csv('y3')
last2 = pd.read_csv('last2.csv')
w30 = www[['item_id','shop_id','item_cnt_month']].loc[www['date_block_num'] == 30]
w6  = pd.merge(w30, last2, on=['shop_id', 'item_id'],how='inner')
print(np.sum(w6['item_cnt_month']!=w6['item_cnt_day']))
w6.to_csv('w6.csv', index = False)
