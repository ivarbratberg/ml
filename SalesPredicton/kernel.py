
# coding: utf-8

# In[9]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import date
from datetime import datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Problem defintion**
# We are asking you to predict total sales for every product and store in the next month. By solving this competition you will be able to apply and enhance your data science skills.
# "Store in the next month" is vage. I intepret it to mean that we should store a number in a column and a row, where row is the product id and column is the next month.

# In[10]:


items_df = pd.read_csv('../input/items.csv')
test_df = pd.read_csv('../input/test.csv')
sales_train_df = pd.read_csv('../input/sales_train.csv')
item_categories_df = pd.read_csv('../input/item_categories.csv')
shops_df = pd.read_csv('../input/shops.csv')
test_df = pd.read_csv('../input/test.csv')


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

# In[11]:


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

returned_items = sales_train_df[sales_train_df.item_cnt_day <0]
returned_dates = returned_items.date.apply(lambda x:datetime.strptime(x,'%d.%m.%Y'))
returned_items['day'] = returned_dates
returned_items = returned_items.sort_values(by=['day'])
plt.plot(returned_items['day'],returned_items.item_cnt_day)
plt.show()

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
    plt.show()


# ## Items data description ##  
# We have about 22170 different items in 84 different categories  
# ** item_category_id **: [0-83]  
# ** item_name **: each line unique => 22170 different values  
# ** item_id **: each line unique   => [0 - 22169]

# In[12]:


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

# In[17]:


last = sales_train_df.groupby(['shop_id','item_id','date_block_num'], as_index=False).agg({"item_cnt_day": "sum"}).sort_values('date_block_num').groupby(['shop_id','item_id'], as_index=False).last()
last2=last[last['date_block_num'] == 33]
answer = pd.merge(test_df,last2,on=['shop_id','item_id'],how='left').fillna(0)[['ID','item_cnt_day']]
answer.columns =  ['ID','item_cnt_month']
answer.to_csv('csv_to_submit.csv', index = False)


# Assuming that we have long trends, and seasonal trende we can try to add month of year as a separate factor  and use
# and use gradient boosting tree
# We go back and 
sales_train_df['month'] = sales_train['date_block_num'] % 12
test['month'] = sales_train['date_block_num'] % 12
# There could also be quarterly years trends
sales_train_df['quarter'] = np.floor(sales_train['month'] % 4)
# There could also be half year trends
sales_train_df['half'] = np.floor(sales_train['month'] % 6)

x_columns=['month','quarter','half','date,date_block_num','shop_id,item_id','item_price']
# Split the data into test and validation
X_train = sales_train_df[x_columns].where(sales_train_df['date_block_num']<last_block)
Y_train = sales_train_df['item_cnt_day'].where(sales_train_df['date_block_num']<last_block )
X_test = sales_train_df[x_columns].where(sales_train_df['date_block_num']==(last_block+1))
Y_test = sales_train_df['item_cnt_day'].where(sales_train_df['date_block_num']==(last_block+1) )

#XGBoost
model = XGBClassifier()
model.fit(X_train, Y_train)
predictions = mode.predict(X_test)
