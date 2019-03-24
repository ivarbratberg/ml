
# coding: utf-8
#%%
# In[9]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



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

global now 
now= time.time()
def rt(prefix):
    global now
    print(prefix + " : " + str(time.time()-now) )
    now = time.time()

last_month_estimate = True
#x_columns=['date_block_num','shop_id','item_id','item_price','days','month','quarter','half','prev_week','yesterday'] # "days"
verbose = False
last_block = 33
returns_study = False
generate_days = False
generate_periods = True
plotboxplot = False

# Any results you write to the current directory are saved as output.


# **Problem defintion**
# We are asking you to predict total sales for every product and store in the next month. By solving this competition you will be able to apply and enhance your data science skills.
# "Store in the next month" is vage. I intepret it to mean that we should store a number in a column and a row, where row is the product id and column is the next month.

# In[10]:


items_df = pd.read_csv('../input/items.csv')
test_df = pd.read_csv('../input/test.csv')
sales_train_df = pd.read_csv('../input/sales_train.csv')
#sales_train_df = pd.read_csv('sales_train_with_days.csv')
sales_train_df.drop(['date'], axis=1)
item_categories_df = pd.read_csv('../input/item_categories.csv')
shops_df = pd.read_csv('../input/shops.csv')
test_df = pd.read_csv('../input/test.csv')
rt("read files ")
# We use this as long as we tune the algorithm
sales_train_df = sales_train_df.loc[sales_train_df['date_block_num']<=last_block]
#sales_train_df = sales_train_df.groupby(['shop_id','item_id','date_block_num'], as_index=False).agg({"item_cnt_day": "sum","item_price": "mean"}).sort_values('date_block_num')

format_str = '%d.%m.%Y' # The format
#returned_dates = returned_items.date.apply(lambda x:datetime.strptime(x,'%d.%m.%Y'))

#%% 

if generate_days:
    sales_train_df['days'] = sales_train_df['date'].apply(lambda x:(datetime.strptime(x,'%d.%m.%Y').date() -date(2013,1,1)).days)
    sales_train_df.to_csv('sales_train_with_days.csv',  index = False)
# ## Describe the data ##
# sales_train:   
#     date:date  
#     date_block_num:[0-33 ]  
#     shop_id:integer describing the shop[0,59]   
#     item_id: has 21807 uniqe values  
#     ** item_cnt_day **: spans nont contonius from -22 to 2169  . ** What means negative numbers ? ** We can try to see how many uniqie items are returned.
#     
#     
#     
#  
#     


if verbose:
    print(sales_train_df.columns.values)
    print(sales_train_df.head())
    print(sales_train_df.tail())
    print(sales_train_df.info())
    print(sales_train_df.describe())
    print(len(np.unique(sales_train_df[['item_id']])))
    print(len(np.unique(sales_train_df[['shop_id']])))
    np.set_printoptions(precision=0, suppress=True)
    print( 'uniqe cnt_day' + str(np.unique(sales_train_df[['item_cnt_day']])))


# ## Negative cnts survey ##
# First we plot the negative counts as function of date
# It does not reveal so much 
# ## The second plot compares sale and return of one specific article ##

if returns_study:
    returned_items = sales_train_df[sales_train_df.item_cnt_day <0]
    returned_dates = returned_items.date.apply(lambda x:datetime.strptime(x,'%d.%m.%Y'))
    returned_items['day'] = returned_dates
    returned_items = returned_items.sort_values(by=['day'])
    plt.plot(returned_items['day'],returned_items.item_cnt_day)
    plt.savefig("returned_items.pdf", bbox_inces='tight')
    #plt.show()

    ixs = returned_items.item_cnt_day.argsort()


# Lets check the most returned item ids, and  plot time history
# We dont see any specific patterns, we can maybe use it as a feature

    for ix in ixs[0:4]:
        tp = returned_items.item_id.iloc[ix]

        tp_frame = sales_train_df[sales_train_df.item_id == tp]
        tp_dates = tp_frame.date.apply(lambda x:datetime.strptime(x,'%d.%m.%Y'))
        tp_frame['day'] = tp_dates
        tp_frame = tp_frame.sort_values('day')
        plt.plot(tp_frame['day'], tp_frame.item_cnt_day)
        plt.savefig("negative " + str(ix) +".pdf", bbox_inces='tight')


# ## Items data description ##  
# We have about 22170 different items in 84 different categories  
# ** item_category_id **: [0-83]  
# ** item_name **: each line unique => 22170 different values  
# ** item_id **: each line unique   => [0 - 22169]

#%%
if verbose:
    print(items_df.columns.values)
    print(items_df.head())
    print(items_df.tail())
    print(items_df.info())
    print(items_df.describe())
    print(items_df.describe(include=['O']))
    print( len(np.unique(items_df[['item_category_id']])))
    print( np.unique(items_df[['item_category_id']]))

    print( len(np.unique(items_df[['item_name']])))
    print( len(np.unique(items_df[['item_id']])))
    print( np.min(items_df[['item_id']]))
    print( np.max(items_df[['item_id']]))


    # ## items_categories 
    # Names of the categories

    # In[13]:


    print(item_categories_df.columns.values)
    print(item_categories_df.head())
    print(item_categories_df.tail())
    print(item_categories_df.info())
    print(item_categories_df.describe())
    #print(len(np.unique(item_categories_df[['item_id']])))
    #print(len(np.unique(item_categories_df[['shop_id']])))


    # ## shops ##  
    # Contains names of the shops  
    # Maybe there is something with the name of the shops, but hard to tell.  
    # Maybe we can sort them by which is most similare  

    # In[14]:


    print(shops_df.columns.values)
    print(shops_df.head())
    print(shops_df.tail())
    print(shops_df.info())
    print(shops_df.describe())
    #print(len(np.unique(shops_df[['item_id']])))
    #print(len(np.unique(shops_df[['shop_id']])))


    # ## test table ##  
    # Is this a list of items sold ?
    # 

    # In[15]:


    print(test_df.columns.values)
    print(test_df.head())
    print(test_df.tail())
    print(test_df.info())
    print(test_df.describe())


# ## Summary of tables ##  
# Seems like sales_train is the only with valuable data.  
# Basically we have item, date and number of items sold and in which shops.
# ### test table ###
# This table 
# 
# ### Quality of data ###
# There are no missing data as in Titanic, except we might miss some dates.  
# #### Outliers ####  
# We can check for outliers by looking at histogram for each series, and removing those most away
# 
# 
# #### Few sales ###
# Seems like for some data there is few sales, 
# we must be able to skip these or at least to 
# 
# ### Restacking of data #### 
# We need to pick out some time series for specific products.
# What we see is that some of the series are very short
# 
# 
# 

# In[16]:


sales_train_df.groupby(['item_id']).size()


# ## Scoring ##  
# How do we do the scoring ?   
# We modify the ** sample_submission ** table.
# 
# ## First attempt, take the latest sale and multiply with 30 !! ##  
# ### test id ordering ###  
# The order of the rows in the submission is not logical. As a consequence we would have to pick the shop id and the item id from the test table in a for like loop.  
# This for loop should then look up in tables that we produce and put in the right numbers.  
#  
#  Back to our simplest strategy, which would be to take the latest sale for a product in a shop.  
#  If the product has not been sold, well, then likely it will be sold the next month either !
#  
#    ** as_index = False ** makes the grouped columns come out as ordinary colums
#   
#   ### Explanation of current query ###
#   -  Sort by date block (month nr) and group by shop-item-month, and sum nr sold items per day. 
#   -  Group the results by shop-item, and pick the last, a small trick is used here as we use the combination of sort-grouby-last to find the grouped row with largest date-block-nr
#   - The results is a table with the latest data block when something was sold, and how much was sold then	, it is store in **last**  
#     shop_id	item_id	date_block_num	item_cnt_day  
#     0	0	30	1	31.0  
#     1	0	31	1	11.0  
# - Now we want to remove all rows which do not have 33 , and assign in to **last2**
# - Then we should do an out join with thest test table, assign to **last3**  
# - Replace the Nan with 0  
# - Doing outer join, gave us too many actually, we are interested in only extra values from left, this is a **left join**
# 
#   
#   
#   
#   
#  
#  

# In[17]:

if last_month_estimate:
    # .sort_values('date_block_num').groupby(['shop_id','item_id'], as_index=False).last()
    last = sales_train_df.groupby(['shop_id','item_id','date_block_num'], as_index=False).agg({"item_cnt_day": "sum"})
    last2=last[last['date_block_num'] == 30]
    last2.to_csv('last2.csv', index=False)
    answer = pd.merge(test_df,last2,on=['shop_id','item_id'],how='left').fillna(0)[['ID','item_cnt_day']]
    answer.columns =  ['ID','item_cnt_month']
    answer['item_cnt_month'][answer['item_cnt_month'] > 20 ] = 20
    answer.to_csv('csv_to_submit.csv', index = False)
    rt("last_month")

if generate_periods:
    # Assuming that we have long trends, and seasonal trende we can try to add month of year as a separate factor  and use
    # and use gradient boosting tree
    # We go back and 
    sales_train_df['month'] = sales_train_df['date_block_num'] % 12
    test_df['month'] = sales_train_df['date_block_num'] % 12
    # There could also be quarterly years trends
    sales_train_df['quarter'] = np.floor(sales_train_df['month'] % 4)
    # There could also be half year trends
    sales_train_df['half'] = np.floor(sales_train_df['month'] % 6)


#%% 
rt("generate periods")
sales_train_df=sales_train_df.drop(columns=['date'])

# ## Now we use days to get week and day lag
# prev_week = sales_train_df[['days', 'shop_id', 'item_id','item_cnt_day','item_price']].copy()
# prev_week = prev_week.rename(columns={'item_cnt_day' : 'item_cnt_prev_week', 'item_price' : 'item_price_prev_week'})

# yesterday = sales_train_df[['days', 'shop_id', 'item_id','item_cnt_day','item_price']].copy()
# yesterday = yesterday.rename(columns={'item_cnt_day' : 'item_cnt_yesterday', 'item_price' : 'item_price_yesterday'})

# yesterday2 = sales_train_df[['days', 'shop_id', 'item_id','item_cnt_day','item_price']].copy()
# yesterday2 = yesterday2.rename(columns={'item_cnt_day' : 'item_cnt_yesterday2', 'item_price' : 'item_price_yesterday2'})

# yesterday3 = sales_train_df[['days', 'shop_id', 'item_id','item_cnt_day','item_price']].copy()
# yesterday3 = yesterday3.rename(columns={'item_cnt_day' : 'item_cnt_yesterday3', 'item_price' : 'item_price_yesterday3'})

# yesterday4 = sales_train_df[['days', 'shop_id', 'item_id','item_cnt_day','item_price']].copy()
# yesterday4 = yesterday4.rename(columns={'item_cnt_day' : 'item_cnt_yesterday4', 'item_price' : 'item_price_yesterday4'})

# yesterday5 = sales_train_df[['days', 'shop_id', 'item_id','item_cnt_day','item_price']].copy()
# yesterday5 = yesterday5.rename(columns={'item_cnt_day' : 'item_cnt_yesterday5', 'item_price' : 'item_price_yesterday5'})


# prev_week['days'] += 7
# yesterday['days'] += 1
# yesterday2['days'] += 2
# yesterday3['days'] += 3
# yesterday4['days'] += 4
# yesterday5['days'] += 5
# w = pd.merge(sales_train_df, prev_week,  on = ['shop_id','item_id','days'], how='left').fillna(0)
# ww = pd.merge(w, yesterday,  on = ['shop_id','item_id','days'], how='left').fillna(0)
# ww = pd.merge(ww, yesterday2,  on = ['shop_id','item_id','days'], how='left').fillna(0)
# ww = pd.merge(ww, yesterday3,  on = ['shop_id','item_id','days'], how='left').fillna(0)
# ww = pd.merge(ww, yesterday4,  on = ['shop_id','item_id','days'], how='left').fillna(0)
# ww = pd.merge(ww, yesterday5,  on = ['shop_id','item_id','days'], how='left').fillna(0)

# ww = ww.drop(columns=['days'])
# rt("ww")

# Check price outlayers
if plotboxplot:
    plt.figure(0)
    plt.boxplot(sales_train_df['item_price'])
    # we should remove > 37000
    plt.figure(1)
    plt.boxplot(sales_train_df['item_cnt_day'])
    plt.show()

# Replace negative and extreme values
median = np.median(sales_train_df['item_price'])
sales_train_df['item_price']=np.where(sales_train_df.item_price < 0, median, sales_train_df.item_price)
sales_train_df['item_price']=sales_train_df['item_price'].clip(0,37000)

# Aggregate over month
www = sales_train_df.groupby(['date_block_num','shop_id','item_id'], as_index=False).agg({'item_cnt_day' : 'sum', 'item_price': 'mean'})
www = www.rename(columns={'item_cnt_day' : 'item_cnt_month'})

# merge in category id
#item_categories_df = pd.read_csv('../input/item_categories.csv')
# shops_df = pd.read_csv('../input/shops.csv')
# items_df = pd.read_csv('../input/items.csv')
www = pd.merge(www, items_df, on="item_id")
#www = pd.merge(www, item_categories_df, on="item_category_id")


www = www.drop(columns=['item_name'])
www.to_csv('y3', index = False)

rt("aggregate over month into www")


    
    


    
