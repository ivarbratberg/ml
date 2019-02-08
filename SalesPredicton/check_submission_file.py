from numpy import loadtxt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import date
import time
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures

my = pd.read_csv('C:/Users/ibratberg/Google Drive/csv_to_submit.csv')
their = pd.read_csv('../sample_submission.csv')
print("så far")