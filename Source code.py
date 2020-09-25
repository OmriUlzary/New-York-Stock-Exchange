#!/usr/bin/env python
# coding: utf-8

# In[311]:


import os
import math
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pandas.plotting import lag_plot
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import statsmodels.stats as sms
import statsmodels.api as sm
from scipy.stats import norm
from numpy.random import normal, seed
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


# In[312]:


def read_data(csv_file):
    try:
        return pd.read_csv(csv_file, index_col='Date', parse_dates=['Date'])
    except:
        print("The file is not found")
        return None

stock_data_set = read_data("C:/Users/omri1/PycharmProjects/untitled2/prices.csv")


# In[313]:


stock_data_set.head()


# In[314]:


stock_data_set.plot(title="New York Stocks")


# In[315]:


stock_data_set[["Low", "High"]].plot(title="New York Stocks: Low vs. High")


# In[316]:


stock_data_set['2011':'2013'][["Low", "High"]].plot(subplots=True) # split the columns to different plots
plt.title('New York stock attributes from 2011 to 2013')
plt.show()


# In[317]:


ax = stock_data_set[["Low"]].plot(color='blue',fontsize=8)
ax.set_xlabel('Date')
ax.set_ylabel('Low')
# add markers
ax.axvspan('2014-01-01','2014-01-31', color='red', alpha=0.3)
ax.axhspan(600, 900, color='black',alpha=0.3)
plt.title("New York Low range in January 2014")
plt.show()


# In[318]:


sns.kdeplot(stock_data_set['Low'], shade=True)
plt.title("Kernel Density Estimator")


# In[319]:


sns.kdeplot(stock_data_set['Volume'], shade=True)
plt.title("Kernel Density Estimator")


# In[320]:


lag_plot(stock_data_set["Volume"]) # lag plot is the dependency of Y(t+1) in Y(t)
plt.title("AutoCorrelation of New York Volume")
plt.show()


# In[321]:


lag_plot(stock_data_set["High"])
plt.title("AutoCorrelation of New York High Stock")
plt.show()


# In[322]:


# Examples for different autoregressive values, That is linearly dependtion on its own previous values.

SAMPLES = 100

def ar_0(size, constant, noise):
    x = np.zeros(size)
    for i in range(size):
        x[i] = constant + noise[i]
    return x

e = np.random.randn(SAMPLES)
x = ar_0(SAMPLES, 0.4, e)

plt.figure(figsize=(10, 4))
plt.plot(range(SAMPLES), x, label="x")
plt.plot(range(SAMPLES), e, label="e")
plt.title("X and E samples using AR(0)")


# In[323]:


def ar_1(size, p, constant, noise):
    x = np.zeros(size)
    for i in range(p, SAMPLES):
        x[i] = constant[0] * x[i-1] + e[i]
    return x

a = [0.5]
p = len(a)
y = ar_1(SAMPLES, len(a), a, e)

plt.figure(figsize=(10, 4))
plt.plot(range(SAMPLES), e, label="e")
plt.plot(range(SAMPLES), y, label="y")
plt.title("E and Y samples using AR(1) and a=0.5")


# In[324]:


a = [0.9]
y = ar_1(SAMPLES, len(a), a, e)
plt.figure(figsize=(10, 4))
plt.plot(range(SAMPLES), e, label="e")
plt.plot(range(SAMPLES), y, label="y")
plt.title("E and Y samples using AR(1) and a=0.9")

def ar_2(size, p, constant, noise):
    x = np.zeros(size)
    for i in range(p, SAMPLES):
        x[i] = constant[0]*x[i-2] + constant[1]*x[i-1] + e[i]
    return x

a = [0.5, 0.5]
y = ar_2(SAMPLES, len(a), a, e)
plt.figure(figsize=(10, 4))
plt.plot(range(SAMPLES), e, label="e")
plt.plot(range(SAMPLES), y, label="y")
plt.title("E and Y samples using AR(2) and a=[0.5, 0.5]")


# In[325]:


a = [-0.5, 0.1]
y = ar_2(SAMPLES, len(a), a, e)
plt.figure(figsize=(10, 4))
plt.plot(range(SAMPLES), e, label="e")
plt.plot(range(SAMPLES), y, label="y")
plt.title("E and Y samples using AR(2) and a=[0.9, 0.1]")


# In[326]:


def moving_average(numbers, N):
    i = 0
    moving_averages = []
    while i < len(numbers) - N + 1: # the chunk of last N observations
        N_tag = numbers[i : i + N]
        window_average = sum(N_tag) / N
        moving_averages.append(window_average)
        i += 1
    return moving_averages

moving_average([1, 2, 4, 5, 7, 9], 3)


# In[327]:


moving_average([1, 1, 1, 1, 1, 1], 3)


# In[328]:


# The Naive Algorithm

X = stock_data_set["High"]
splitter = int(len(X) * 0.7)
train, test = X[:splitter], X[splitter:]

g_high = train.to_numpy()
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, [train[len(train)-1]] * len(test), label="Forecast")
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.show()


# In[335]:


# PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

stock_data_set


# In[340]:


stock_data_set.drop(columns=["Symbol"], inplace=True)

sc = StandardScaler()
normalized_data = sc.fit_transform(stock_data_set)
pca = PCA()
pca_data = pca.fit_transform(normalized_data)


# In[341]:


plt.bar(range(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Components')
plt.ylabel('Cumulative Variance')
plt.xticks(range(1, len(pca.explained_variance_ratio_)+1))
plt.title("Components Variance")
plt.plot()


# In[342]:


pd.DataFrame({
"Variance": pca.explained_variance_ratio_
}, index=range(1, len(pca.explained_variance_ratio_) + 1))


# In[343]:


# using components = 1
pca = PCA(n_components=1)
pca_data = pca.fit_transform(normalized_data)
components = pd.DataFrame(pca.components_, columns = stock_data_set.columns)
components


# In[344]:


pd.DataFrame(pca_data).plot(title="1D New York Stock", figsize=(10,5))

