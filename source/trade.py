from info import *
from request import send_signed_request
import time
import numpy as np
from config import api_key, api_secret, SYMBOL
from orders import *

INTERVAL = 10
MARGIN = 3
DEPTH = 5
MAX_ORDER = 5
POSITION_LIMIT = 0.05
weight = [0.35, 0.25, 0.15, 0.10, 0.05]
cancel_price_diff = 0.00002
tick_size = 0.01
global price_precision
#  "minPrice": "0.00000100",
#  "maxPrice": "100000.00000000",
#  "tickSize": "0.00000100"


# symbol_info = get_symbol_info(SYMBOL)
# price_precision = int(symbol_info['quoteAssetPrecision'])
# print(price_precision) # 8
# TODO:don't know why only price_precision <= 1/2?  works
price_precision = 1
# tick_size = float(symbol_info['filters'][0]['tickSize'])
# print(tick_size)  # 0.01

# min_qty, max_qty, step_size = get_symbol_limits(symbol_info)
# print(min_qty, max_qty, step_size)  # 1e-05 9000.0 1e-05


global quantity_precision
quantity_precision = 5


def get_position_size():
    endpoint = '/fapi/v2/positionRisk'
    params = {
        'symbol': SYMBOL,
    }
    response = send_signed_request(
        'GET', endpoint, api_key, api_secret, params)
    if response:
        position_size = float(response[0]['positionAmt'])
    else:
        position_size = 0
    return position_size


def get_account_information():
    endpoint = '/fapi/v2/account'
    response = send_signed_request('GET', endpoint, api_key, api_secret)
    return response


def get_account_trade_list():
    endpoint = '/fapi/v1/userTrades'
    params = {
        'symbol': SYMBOL,
    }
    response = send_signed_request(
        'GET', endpoint, api_key, api_secret, params)
    return response


# TODO: return nothing[]
# get_account_trade_list()

def get_ticker_price():
    ''' get the latest ticker price for a specific symbol'''
    endpoint = '/fapi/v1/ticker/price'
    params = {
        'symbol': SYMBOL,
    }
    response = send_signed_request(
        'GET', endpoint, api_key, api_secret, params)
    return response


def get_klines_data():
    endpoint = '/fapi/v1/klines'
    params = {
        'symbol': SYMBOL,
        'interval': '1h',  # Available intervals: '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'
    }
    response = send_signed_request(
        'GET', endpoint, api_key, api_secret, params)
    return response


def get_order_book_depth():
    endpoint = '/fapi/v1/depth'
    params = {
        'symbol': SYMBOL,
        'limit': DEPTH,
    }
    response = send_signed_request(
        'GET', endpoint, api_key, api_secret, params)
    return response


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


class AutoTrader():

    def __init__(self):
        """Initialise a new instance of the AutoTrader class."""
        self.bids = []
        self.asks = []
        self.bids_hedge = set()
        self.asks_hedge = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0

    def work(self):
        order_book_data = get_order_book_depth()
        self.bids = order_book_data['bids']
        self.asks = order_book_data['asks']
        self.position = get_position_size()
        # print('Bids:', self.bids)
        # print('Asks:', self.asks)
        print('Position:', self.position)

        new_ask_price = round(calc_vwap(self.asks), price_precision)
        new_bid_price = round(calc_vwap(self.bids), price_precision)

        # if(new_ask_price <= new_bid_price):
        #     return

        print("new_ask_price:", new_ask_price, "new_bid_price:", new_bid_price)
        # new_ask_price += MARGIN
        # new_bid_price -= MARGIN

        should_notify_open_long = False
        should_notify_open_short = False

        if np.abs(self.bid_price - new_bid_price) > self.bid_price * cancel_price_diff and self.position < POSITION_LIMIT:
            should_notify_open_long = True

        if np.abs(self.ask_price - new_ask_price) > self.ask_price * cancel_price_diff and self.position > -POSITION_LIMIT:
            should_notify_open_short = True

        # bid
        if new_bid_price > 0 and should_notify_open_long:
            quantity = round(min(MAX_ORDER, POSITION_LIMIT -
                                 self.position), quantity_precision)
            # quantity = 100
            new_bid_price = round(new_bid_price, price_precision)

            send_cancel_order(self.bid_id)
            send_cancel_order(self.ask_id)
            self.bid_price = new_bid_price
            self.bid_id = place_limit_order(SYMBOL, 'BUY', 'LONG',
                                            quantity, new_bid_price)

        # ask
        if new_ask_price > 0 and should_notify_open_short:
            quantity = round(min(MAX_ORDER, POSITION_LIMIT +
                             self.position), quantity_precision)
            # quantity = 100
            new_ask_price = round(new_bid_price, 1)
            send_cancel_order(self.bid_id)
            send_cancel_order(self.ask_id)
            self.ask_price = new_ask_price
            self.ask_id = place_limit_order(
                SYMBOL, 'SELL', 'SHORT', quantity, new_ask_price)


autotrader = AutoTrader()
# place_market_order(SYMBOL, 'SELL', 'SHORT', 5)
# place_market_order(SYMBOL, 'BUY', 'LONG', 5)
while(True):
    autotrader.work()
    time.sleep(INTERVAL)
