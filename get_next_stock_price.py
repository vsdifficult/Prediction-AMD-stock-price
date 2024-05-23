import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv('AMD.csv')

data = dataset.filter(['Close']).values
data = data.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


def create_dataset(data, time_step):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)


time_step = 60
X, Y = create_dataset(scaled_data, time_step)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(units=150, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=150, return_sequences=True))
model.add(LSTM(units=150, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, Y, epochs=150, batch_size=32)

last_inputs = scaled_data[-time_step:]
last_inputs = last_inputs.reshape(1, -1)
last_inputs = np.reshape(last_inputs, (1, time_step, 1))
predicted_price = model.predict(last_inputs)
predicted_price = scaler.inverse_transform(predicted_price)

with open("predict.txt", 'w', encoding='utf-8') as file:
    file.write(str(predicted_price[0][0]))
