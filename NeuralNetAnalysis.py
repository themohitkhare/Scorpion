'''
Run the file with THEANO_FLAGS=device=cuda python using the terminal
'''
import theano.gpuarray
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
import keras
import pandas_datareader.data as web
import h5py
from keras import backend as K
import quandl

quandl.ApiConfig.api_key = 'tHhnk2LX1KrKyxKKyhaz'
seq_len = 22
shape = [seq_len, 9, 1]
neurons = [256, 256, 32, 1]
dropout = 0.3
decay = 0.5
epochs = 72
stock_code = '500112'


def get_stock_data(stock_code, normalize=True, ma=[]):
    """
    Return a dataframe of that stock and normalize all the values.
    (Optional: create moving average)
    """
    df = quandl.get('BSE/BOM' + stock_code)
    df.drop(
        ['WAP', 'No. of Trades', 'Total Turnover', 'Deliverable Quantity', '% Deli. Qty to Traded Qty', 'Spread H-L',
         'Spread C-O'], 1, inplace=True)
    df['date'] = df.index
    df.set_index('date', inplace=True)

    # Renaming all the columns so that we can use the old version code
    df.rename(columns={'No. of Shares': 'Volume', 'Close': 'Adj Close'}, inplace=True)

    # Percentage change
    df['Pct'] = df['Adj Close'].pct_change()
    df.dropna(inplace=True)

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
        df['Pct'] = min_max_scaler.fit_transform(df['Pct'].values.reshape(-1, 1))
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

corr = df.corr()
ax = sns.heatmap(corr, cmap="YlGnBu")
plt.show()


def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    print("Amount of features = {}".format(amount_of_features))
    data = stock.as_matrix()
    sequence_length = seq_len + 1  # index starting from 0
    result = []

    for index in range(len(data) - sequence_length):  # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length])  # index : index + 22days

    result = np.array(result)
    print(result.shape)
    row = round(0.8 * result.shape[0])  # 80% split
    print("Amount of training data = {}".format(0.9 * result.shape[0]))
    print("Amount of testing data = {}".format(0.1 * result.shape[0]))

    train = result[:int(row), :]  # 90% date
    X_train = train[:, :-1]  # all data until day m
    y_train = train[:, -1][:, -1]  # day m + 1 adjusted close price

    X_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

    return [X_train, y_train, X_test, y_test]


X_train, y_train, X_test, y_test = load_data(df, seq_len)


def build_model(shape, neurons, dropout, decay):
    model = Sequential()

    model.add(LSTM(neurons[0], input_shape=(shape[0], shape[1]), return_sequences=True))
    model.add(Dropout(dropout))

    model.add(LSTM(neurons[1], input_shape=(shape[0], shape[1]), return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(neurons[2], kernel_initializer="uniform", activation='relu'))
    model.add(Dense(neurons[3], kernel_initializer="uniform", activation='linear'))
    # model = load_model('my_LSTM_stock_model1000.h5')
    adam = keras.optimizers.Adam(decay=decay)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


model = build_model(shape, neurons, dropout, decay)


model.fit(
    X_train,
    y_train,
    batch_size=512,
    epochs=epochs,
    validation_split=0.2,
    verbose=1)


def model_score(model, X_train, y_train, X_test, y_test):
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    return trainScore[0], testScore[0]


model_score(model, X_train, y_train, X_test, y_test)


def percentage_difference(model, X_test, y_test):
    percentage_diff = []

    p = model.predict(X_test)
    for u in range(len(y_test)):  # for each data index in test data
        pr = p[u][0]  # pr = prediction on day u

        percentage_diff.append((pr - y_test[u] / pr) * 100)
    return p


p = percentage_difference(model, X_test, y_test)


def denormalize(stock_code, normalized_value):
    """
    Return a dataframe of that stock and normalize all the values.
    (Optional: create moving average)
    """
    df = quandl.get('BSE/BOM' + stock_code)
    df.drop(
        ['WAP', 'No. of Trades', 'Total Turnover', 'Deliverable Quantity', '% Deli. Qty to Traded Qty', 'Spread H-L',
         'Spread C-O'], 1, inplace=True)
    df['date'] = df.index
    df.set_index('date', inplace=True)

    # Renaming all the columns so that we can use the old version code
    df.rename(columns={'No. of Shares': 'Volume', 'Close': 'Adj Close'}, inplace=True)

    df.dropna(inplace=True)
    df = df['Adj Close'].values.reshape(-1, 1)
    normalized_value = normalized_value.reshape(-1, 1)

    # return df.shape, p.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(df)
    new = min_max_scaler.inverse_transform(normalized_value)

    return new


def plot_result(stock_name, normalized_value_p, normalized_value_y_test):
    newp = denormalize(stock_name, normalized_value_p)
    newy_test = denormalize(stock_name, normalized_value_y_test)
    plt2.plot(newp, color='red', label='Prediction')
    plt2.plot(newy_test, color='blue', label='Actual')
    plt2.legend(loc='best')
    plt2.title('The test result for {}'.format(stock_name))
    plt2.xlabel('Days')
    plt2.ylabel('Adjusted Close')
    plt2.show()


plot_result(stock_code, p, y_test)
