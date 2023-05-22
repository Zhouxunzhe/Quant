from function import *
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
import torch.nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from matplotlib import pyplot as plt


def calc_vwap(order_book):
    A = 0
    B = 0
    for i in range(5):
        price = float(order_book[i][0])
        volume = float(order_book[i][1])
        A += weight[i] * price * volume
        B += weight[i] * volume
    if B < 0.001:  # 0.001 for BTCUDST
        return 0
    return A / B


def calc_buy_point():
    size = len(get_klines_data())
    kline_data = np.array(get_klines_data()).astype('float64')
    change_vector = np.diff(kline_data[:, 4])
    close_data = np.where(change_vector > 0, 1.0, 0.0)
    factors = np.delete(kline_data, [0, 6, 11], axis=1)
    train_data = factors[0:int(size * 0.9), :]
    train_label = close_data[0:int(size * 0.9)]
    test_data = factors[int(size * 0.9):size - 1, :]
    test_label = close_data[int(size * 0.9):]
    # params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
    # knn = neighbors.KNeighborsRegressor()
    # model = GridSearchCV(knn, params, cv=5)
    # model = RidgeClassifier()
    model = SVC()
    model.fit(train_data, train_label)
    prediction = model.predict(test_data)
    prediction = np.where(prediction > 0.5, 1.0, 0.0)

    precisions = np.sum(prediction == test_label) / len(prediction)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(torch.tensor(prediction), torch.tensor(test_label))
    print("precision = ", precisions)

    buy_point = []
    return buy_point


def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        X.append(dataset[i:(i+look_back), :])
        Y.append(dataset[i+look_back+1, 3])
    return np.array(X), np.array(Y)


def calc_buy_point_lstm():
    size = len(get_klines_data(100))
    kline_data = np.array(get_klines_data(100)).astype('float64')
    data = np.delete(kline_data, [0, 5, 6, 7, 8, 9, 10, 11], axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = scaler.fit_transform(data)
    test_data = train_data[size-32:, :]

    look_back = 30
    trainX, trainY = create_dataset(train_data, look_back)
    testX, testY = create_dataset(test_data, look_back)

    model = Sequential()
    model.add(LSTM(32, input_shape=(look_back, train_data.shape[1])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=10, batch_size=1)
    testPredict = model.predict(testX)

    buy_point = False
    print(data[size-1:, 3])
    if testPredict > testY:
        buy_point = True

    return buy_point
