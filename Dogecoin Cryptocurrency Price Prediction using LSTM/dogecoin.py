# -*- coding: utf-8 -*-
"""Dogecoin.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1twWZxPiG5YY3Mon8dtZgKQWMLepRMbN1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

dt=pd.read_csv('/content/DOGE-USD.csv')
dt

dt.head()

dt=dt[["Close"]]
dt.fillna(method="ffill", inplace=True)
scaler=MinMaxScaler()
dt=scaler.fit_transform(dt)

train_len= int(len(dt) * 0.8)
train, test= dt[0:train_len], dt[train_len:]
def seqcreate(dt, look_back=1):
  X,Y = [],[]
  for i in range(len(dt) - look_back - 1):
    X.append(dt[i:(i+look_back), 0])
    Y.append(dt[i+look_back, 0])
  return np.array(X), np.array(Y)

look_back = 1
X_train, Y_train = seqcreate(train, look_back)
X_test, Y_test = seqcreate(test, look_back)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")

model.fit(X_train,Y_train, epochs=150, batch_size=32, validation_data=(X_test, Y_test))

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
origprices = scaler.inverse_transform(Y_test.reshape(-1,1))
plt.plot(origprices, label= 'Original Prices', color='blue')
plt.plot(predictions, label= 'Predicted Prices', color='red')

plt.title('Dogecoin Cryptocurrency Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

rmse = np.sqrt(np.mean((predictions - Y_test) ** 2))
mae = np.mean(abs(predictions - Y_test))
print("RMSE is:", rmse)
print("MAE is:", mae)