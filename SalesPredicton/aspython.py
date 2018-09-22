# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

items_df = pd.read_csv('../input/items.csv')
test_df = pd.read_csv('../input/test.csv')
sales_train_df = pd.read_csv('../input/sales_train.csv')
item_categories_df = pd.read_csv('../input/item_categories.csv')
shops_df = pd.read_csv('../input/shops.csv')
test_df = pd.read_csv('../input/test.csv')
