from function import *
import numpy as np
import torch.nn as nn
import torch
from sklearn.ensemble import RandomForestRegressor


def swing_rf():
    size = 1000
    pred_size = 100
    kline_data = np.array(get_klines_data(size)).astype('float64')
    change_vector = np.diff(kline_data[:, 4])
    close_data = np.where(change_vector > 0, 1.0, 0.0)
    factors = np.delete(kline_data, [0, 6, 11], axis=1)
    train_data = factors[0:size - pred_size, :]
    train_label = close_data[0:size - pred_size]
    test_data = factors[size - pred_size:size - 1, :]
    test_label = close_data[size - pred_size:]
    model = RandomForestRegressor(random_state=44, max_depth=10, max_features='sqrt', min_samples_leaf=4,
                                  min_samples_split=2, n_estimators=70)
    model.fit(train_data, train_label)
    prediction = model.predict(test_data)
    prediction = np.where(prediction > 0.5, 1.0, 0.0)

    precisions = np.sum(prediction == test_label) / len(prediction)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(torch.tensor(prediction), torch.tensor(test_label))
    print("random forest precision = ", precisions)
