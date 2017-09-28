import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

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
plt.plot(troot, color="red")
plt.plot(predict, color='blue')
plt.show()
