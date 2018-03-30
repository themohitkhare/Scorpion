"""
The module contains modules that enable prediction of indian stock market based on news analysis of the company.
This modules follows Neural Network approach, hence training and prediction takes time.
A single company can take upto 40 mins on CPU and around 5 to 10 mins based on the NVIDIA GPU.
The prediction in stored in the form of graphs that contains both real values and the predicted values.
If using theano
    *Run the file with THEANO_FLAGS=device=cuda python using the terminal for GPU.
    *Run the file with THEANO_FLAGS=device=cpu python using the terminal for CPU.
    
"""

import math
import os
import sqlite3

import keras
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np
import pandas as pd
import quandl
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, model_from_json
from sklearn import preprocessing



quandl.ApiConfig.api_key = 'tHhnk2LX1KrKyxKKyhaz'
seq_len = 21
days = 365
s_name = ''
shape = [seq_len, 9, 1]
neurons = [256, 256, 32, 1]
dropout = 0.3
decay = 0.5
epochs = 75


def get_stock_data(stock_code, normalize=True, ma=[]):
    """
    Method takes BSE stock code, normalize bollean and a list for mooving averages.
    Collects stock information from Quandl using an API call.
    Removes unnecessary information and normalises the required data for training and testing.

    :type stock_code: String
    :param stock_code: Stock code of comapny in BSE
    :type normalize: Boolean
    :param normalize: normalise the values or not
    :type ma: list
    :param ma: List of moving averages
    :return df: DataFrame
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


def plot_stock(df):
    """
    Method used for visual aids for evaluation of Closing rates and percentage chage of a particular company during
    a fixed interval.

    :type df: pandas.DataFrame
    :param df: DataFrame
    :return None:
    """
    print(df.tail())
    plt.subplot(211)
    plt.plot(df['Adj Close'], color='red', label='Adj Close')
    plt.legend(loc='best')
    plt.subplot(212)
    plt.plot(df['Pct'], color='blue', label='Percentage change')
    plt.legend(loc='best')
    plt.show(block=False)


def load_data(stock, seq_len):
    """
    Methods takes in a dataframe and prepare the data for training and testing purposes.
    For maximum efficiency we are using 99% data for training and the rest for testing.

    :type stock: pandas.DataFrame
    :param stock: dataframe containing stock data
    :type seq_len: int
    :param seq_len: Number of days for training
    :return [X_train, y_train, X_test, y_test]: List
    """
    amount_of_features = len(stock.columns)
    print("Amount of features = {}".format(amount_of_features))
    data = stock.as_matrix()
    sequence_length = seq_len + 1  # index starting from 0
    result = []

    for index in range(len(data) - sequence_length):  # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length])  # index : index + 22days

    result = np.array(result)
    row = round(0.99 * result.shape[0])  # 80% split
    print("Amount of training data = {}".format(0.99 * result.shape[0]))
    print("Amount of testing data = {}".format(0.01 * result.shape[0]))

    train = result[:int(row), :]  # 90% date
    X_train = train[:, :-1]  # all data until day m
    y_train = train[:, -1][:, -1]  # day m + 1 adjusted close price

    X_test = result[-days:, :-1]
    y_test = result[-days:, -1][:, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

    return [X_train, y_train, X_test, y_test]


def build_model(shape, neurons, dropout, decay):
    """
    Methods creates a models based on fine tuned shapes and fuctions to produce the best result for prediction.
    The Model uses
        *2 LSTM layers
        *2 Dense layers

    :type shape: list
    :param shape: Shape of the data matrix
    :type neurons: list
    :param neurons: List of neurons
    :type dropout: float
    :param dropout: Dropuot to prevent overfitting
    :type decay: float
    :param decay: Decay rate
    :return model: keras.models.Sequential
    """
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


def model_score(model, X_train, y_train, X_test, y_test):
    """
    Methods calulates the training and testing score based on the MSE(Mean Square Error).

    :type model: keras.models.Sequential
    :param model: Model for training
    :type X_train: numpy.array
    :param X_train: Traning Data
    :type y_train: numpy.array
    :param y_train: Training Results
    :type X_test: numpy.array
    :param X_test: Testing Data
    :type y_test: numpy.array
    :param y_test: Testing Results
    :return trainScore, TestScore: Float, Float
    """
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    return trainScore[0], testScore[0]


def percentage_difference(model, X_test, y_test):
    """
    Method predicts the stock values and calculates the percent difference of the prediction Vs real values.

    :type model: keras.models.Sequential
    :param model: Model for training
    :param X_test: Training Result
    :type X_test: numpy.array
    :param y_test: Testing Result
    :type y_test: numpy.array
    :return p: Predicted Values
    """
    percentage_diff = []

    p = model.predict(X_test)
    for u in range(len(y_test)):  # for each data index in test data
        pr = p[u][0]  # pr = prediction on day u

        percentage_diff.append((pr - y_test[u] / pr) * 100)
    return p


def denormalize(stock_code, normalized_value):
    """
    Method denormalises the previousy normalised value for again ploting in the graph.

    :type stock_code: str
    :param stock_code: Stock code of comapny in BSE
    :type normalized_value: numpy.array
    :param normalized_value: Normalised data
    :return new: numpy.array
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


def plot_result(stock_name, normalized_value_p, normalized_value_y_test, s_name):
    """
    Method saves the result in the form of a picture after adjusting the predicted values on the basis of the result
    of news analysis done on the comapny.

    :type stock_name: str
    :param stock_name: Stock code of comapny in BSE
    :type normalized_value_p: numpy.array
    :param normalized_value_p: Normalised Predicted values
    :type normalized_value_y_test: Normalised Test values
    :param normalized_value_y_test: numpy.array
    :type s_name: str
    :param s_name: Name of the comapany
    :return: None
    """
    dataBase = sqlite3.connect('Sting.db')
    db = dataBase.cursor()
    newp = denormalize(stock_name, normalized_value_p)
    newy_test = denormalize(stock_name, normalized_value_y_test)
    query = "SELECT EVALUATION FROM Company_News WHERE NAME IS '{}'".format(s_name)
    db.execute(query)
    try:
        Prob = db.fetchall()[0][0]
        db.execute("SELECT {} FROM Company_News".format(Prob))
        value = db.fetchall()[0][0]
    except IndexError:
        Prob = "Positive"
        value = 0
    if Prob == 'Positive':
        newp *= (1 + (value / 20))
    else:
        newp *= (1 - (value / 20))
    plt2.plot(newp, color='red', label='Prediction')
    plt2.plot(newy_test, color='blue', label='Actual')
    plt2.legend(loc='best')
    plt2.title('The test result for {}'.format(stock_name))
    plt2.xlabel('Days')
    plt2.ylabel('Adjusted Close')
    path = os.getcwd() + "\\Graphs\\" + s_name + ".png"
    plt2.savefig(path, dpi=1080)
    plt2.close()


def StockPrediction():
    """
    Driver program for the entire Moduele
    :return: None
    """
    dataBase = sqlite3.connect('Sting.db')

    db = dataBase.cursor()

    db.execute(
        "SELECT A.Name, A.BSE , B.Frequency FROM StockCode AS A LEFT JOIN Companies AS B WHERE A.Name = B.Name ORDER BY B.Frequency DESC;")

    x = db.fetchall()

    for y in x:

        stock_code = str(y[1])
        print(y[0])
        df = get_stock_data(stock_code, ma=[50, 100, 200])

        X_train, y_train, X_test, y_test = load_data(df, seq_len)

        model = build_model(shape, neurons, dropout, decay)

        d = os.getcwd() + "/Models/"

        if os.path.isfile(d + "{}.json".format(stock_code)):
            # load json and create model
            json_file = open(d + '{}.json'.format(stock_code), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights(d + "{}.h5".format(stock_code))
            model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
            print("Loaded model from disk")
            print("Model Found!!")
        else:
            model.fit(X_train, y_train, batch_size=512, epochs=epochs, validation_split=0.2, verbose=1)
            # Saving
            # serialize model to JSON
            model_json = model.to_json()
            d = os.getcwd() + "/Models/"
            with open(d + "{}.json".format(stock_code), "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(d + "{}.h5".format(stock_code))
            print("Saved model to disk")

        model_score(model, X_train, y_train, X_test, y_test)

        p = percentage_difference(model, X_test, y_test)

        plot_result(stock_code, p, y_test, y[0])
        dataBase.close()
