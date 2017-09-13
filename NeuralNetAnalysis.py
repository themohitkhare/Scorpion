# import seaborn as sns
# import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import pandas as pd
# from pandas import datetime
# import math, time
# import itertools
from sklearn import preprocessing
# import datetime
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras.layers.recurrent import LSTM
# from keras.models import load_model
# import keras
# import pandas_datareader.data as web
# import h5py
# from keras import backend as K
import quandl

quandl.ApiConfig.api_key = 'zpFWg7jpwtBPmzA8sT2Z'
seq_len = 22
shape = [seq_len, 9, 1]
neurons = [256, 256, 32, 1]
dropout = 0.3
decay = 0.5
epochs = 90
stock_code = '500010'


def get_stock_data(stock_code, normalize=True, ma=[]):
    """
    Return a dataframe of that stock and normalize all the values.
    (Optional: create moving average)
    """
    df = quandl.get('BSE/BOM' + stock_code)
    df.drop(
        ['WAP', 'No. of Trades', 'Total Turnover', 'Deliverable Quantity', '% Deli. Qty to Traded Qty', 'Spread H-L',
         'Spread C-O'], 1, inplace=True)
    df.set_index('Date', inplace=True)

    # Renaming all the columns so that we can use the old version code
    df.rename(columns={'No. of Shares': 'Volume', 'Close': 'Adj Close'}, inplace=True)

    # Percentage change
    # df['Pct'] = df['Adj Close'].pct_change()
    # df.dropna(inplace=True)

    # Moving Average
    if ma != []:
        for moving in ma:
            df['{}ma'.format(moving)] = df['Adj Close'].rolling(window=moving).mean()
    df.dropna(inplace=True)

    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1, 1))
        df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1, 1))
        df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1, 1))
        df['Volume'] = min_max_scaler.fit_transform(df.Volume.values.reshape(-1, 1))
        df['Adj Close'] = min_max_scaler.fit_transform(df['Adj Close'].values.reshape(-1, 1))
        # df['Pct'] = min_max_scaler.fit_transform(df['Pct'].values.reshape(-1, 1))
        if ma != []:
            for moving in ma:
                df['{}ma'.format(moving)] = min_max_scaler.fit_transform(
                    df['{}ma'.format(moving)].values.reshape(-1, 1))

                # Move Adj Close to the rightmost for the ease of training
    adj_close = df['Adj Close']
    df.drop(labels=['Adj Close'], axis=1, inplace=True)
    df = pd.concat([df, adj_close], axis=1)

    return df


df = get_stock_data(stock_code, ma=[50, 100, 200])


def plot_stock(df):
    print(df.head())
    plt.subplot(211)
    plt.plot(df['Adj Close'], color='red', label='Adj Close')
    plt.legend(loc='best')
    plt.subplot(212)
    plt.plot(df['Pct'], color='blue', label='Percentage change')
    plt.legend(loc='best')
    plt.show()


plot_stock(df)
