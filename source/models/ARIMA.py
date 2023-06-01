from function import *
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from statsmodels.tsa import stattools
import warnings
warnings.filterwarnings("ignore")


def price_arima():
    size = 500
    pred_size = 50
    kline_data = np.array(get_klines_data(size)).astype('float64')
    data = np.delete(kline_data, [0, 6, 11], axis=1)
    train_data = data[:size-pred_size, :]
    test_data = data[size-pred_size:, :]

    closing_price = train_data[:, 3]
    test_ts = pd.DataFrame(test_data[:, 3])

    ts = pd.DataFrame(closing_price)
    # plot_acf(ts, use_vlines=True, lags=30)

    # 不进行差分
    # D_ts = ts
    # 进行一阶差分
    D_ts = ts.diff().dropna()
    # 进行二阶差分
    # D_ts = ts.diff().diff().dropna()
    # 绘制差分后时序图
    #     # D_ts.plot()

    # plot_acf(D_ts, use_vlines=True, lags=30)
    print('原始序列的ADF检验结果为：', stattools.adfuller(ts))
    print('一阶差分序列的ADF检验结果为：', stattools.adfuller(D_ts))

    LjungBox = stattools.q_stat(stattools.acf(ts)[1:12], len(ts))
    print('原始序列白噪声检验结果为：', LjungBox[1][-1])
    LjungBox = stattools.q_stat(stattools.acf(D_ts)[1:12], len(D_ts))
    print('一阶差分序列白噪声检验结果为：', LjungBox[1][-1])

    model = sm.tsa.ARIMA(ts, order=(2, 1, 2))
    result = model.fit()
    result.summary()

    predict = pd.DataFrame(result.forecast(pred_size))
    predict = predict.reset_index(drop=True)
    plt.plot(test_ts[1:], linewidth=3, alpha=0.8)
    plt.plot(predict[1:], linewidth=1.2)
    plt.show()
