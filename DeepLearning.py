import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout

def square():
    root = []
    value = []
    troot, tvalue = [], []
    for x in range(2001):
        value.append(x)
        root.append(x ** 0.5)
    for x in range(2001, 4000, 1):
        troot.append(x)
        tvalue.append(x ** 0.5)

    df = pd.DataFrame()
    df['X'] = value
    df['Y'] = root
    data = df.as_matrix()

    X_train = data[:, 0]
    Y_train = data[:, 1]

    model = Sequential()
    model.add(Dense(200, input_dim=1, activation='relu'))
    model.add(Dropout(1.245))
    model.add(Dense(200))
    model.add(Dropout(1.245))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    model.fit(Y_train, X_train, epochs=25, batch_size=1)

    predict = model.predict(tvalue)
    print(predict)
    plt.plot(troot, color="red")
    plt.plot(predict, color='blue')
    plt.show()

import sqlite3

dataBase = sqlite3.connect('Sting.db')

db = dataBase.cursor()

db.execute("SELECT A.Name, A.BSE , B.Frequency FROM StockCode AS A LEFT JOIN Companies AS B WHERE A.Name = B.Name ORDER BY B.Frequency DESC;")

x = db.fetchall()

for y in x:
    print(y[1])