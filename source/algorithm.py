from function import *
import numpy as np
from sklearn.svm import SVC
import torch.nn as nn
import torch


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
    factors = np.delete(kline_data, [1, 4, 6, 11], axis=1)
    train_data = factors[0:int(size * 0.9), :]
    train_label = close_data[0:int(size * 0.9)]
    test_data = factors[int(size * 0.9):size-1, :]
    test_label = close_data[int(size * 0.9):]
    model = SVC()
    model.fit(train_data, train_label)
    prediction = model.predict(test_data)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(torch.tensor(prediction), torch.tensor(test_label))
    print("loss = ", loss)

    buy_point = []
    return buy_point
