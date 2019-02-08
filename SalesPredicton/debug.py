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
use_tree= False
use_linear = False
use_prev_month = True
use_linear_meta = False

global now 
now= time.time()
def rt(prefix):
    global now
    print(prefix + " : " + str(time.time()-now) )
    now = time.time()

last_block = 32
w5 = pd.read_csv('y5.csv')

t = w5[(w5['item_cnt_next_month'] == 0) & (w5['date_block_num'] == 32)]
print("jk")