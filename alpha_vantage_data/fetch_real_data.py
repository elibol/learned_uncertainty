from urllib import request
import re
import json
import csv
import os
import pickle
import numpy as np
import datetime
import time


# Documentation: https://www.alphavantage.co/documentation/
api_key = "QCI3RT39F1VOUR21"
tech_stocks = ["MSFT", "AAPL", "GOOGL", "AMZN", "FB", "NFLX", "IBM", "TSLA"]
DJIA = ["MMM", "AXP", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DIS", "DWDP", "XOM",
        "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "MCD", "MRK", "MSFT", "NKE", "PFE",
        "PG", "TRV", "UTX", "UNH", "VZ", "V", "WMT", "WBA"]
SP500 = ["AAPL", "ABBV", "ABT", "ACN", "AGN", "AIG", "ALL", "AMGN", "AMZN", "AXP", "BA", "BAC", "BIIB",
        "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT", "CELG", "CHTR", "CL", "CMCSA", "COF", "COP",
        "COST", "CSCO", "CVS", "CVX", "DHR", "DIS", "DUK", "DWDP", "EMR", "EXC", "F", "FB", "FDX", "FOX",
        "FOXA", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HAL", "HD", "HON", "IBM", "INTC", "JNJ",
        "JPM", "KHC", "KMI", "KO", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "MMM", "MO",
        "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "OXY", "PEP", "PFE", "PG", "PM", "PYPL",
        "QCOM", "RTN", "SBUX", "SLB", "SO", "SPG", "T", "TGT", "TXN", "UNH", "UNP", "UPS", "USB", "UTX",
        "V", "VZ", "WBA", "WFC", "WMT", "XOM"]

stocks = DJIA

def daily_value_estimate(daily_value_dict):
    open_val = float(daily_value_dict["1. open"])
    high_val = float(daily_value_dict["2. high"])
    low_val = float(daily_value_dict["3. low"])
    close_val = float(daily_value_dict["4. close"])
    volume = float(daily_value_dict["5. volume"])
    return close_val


def get_time_series(symbol):
    url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=" + symbol + "&apikey=" + api_key
    htmltext = request.urlopen(url)
    content = htmltext.read()
    data = json.loads(content)
    dates = np.array([])
    result = np.array([])
    try:
        time_dict = data["Time Series (Daily)"]
        dates = time_dict.keys()
        dates = sorted(dates, key=lambda date_str: datetime.datetime.strptime(date_str, "%Y-%m-%d").timestamp())
        result = np.zeros(shape=(len(dates),))
        for i in range(result.shape[0]):
            result[i] = daily_value_estimate(time_dict[dates[i]])
    except Exception as e:
        print(symbol, "failed:", e, url)
    # Throttled at 5 requests per minute.
    time.sleep(12.0)
    return dates, result

all_data = []
all_dates = []
for stock in stocks:
    single_dates, single_data = get_time_series(stock)
    print(stock, single_data.shape)
    all_data.append(single_data)
    all_dates.append(single_dates)


all_data = np.array(all_data).T
print("num_samples=", all_data.shape[0])
print("num_assets=", all_data.shape[1])

# ensure dates are the same
for i in range(all_data.shape[1]):
    for j in range(i, all_data.shape[1]):
        assert all_dates[i] == all_dates[j]
print("date range:", all_dates[0][0], "-", all_dates[0][-1])

with open("./data/real_data_symbols.pickle", "wb") as fh:
    pickle.dump(stocks, fh)

with open("./data/real_data_dates.pickle", "wb") as fh:
    pickle.dump(all_dates, fh)

with open("./data/real_data.pickle", "wb") as fh:
    pickle.dump(all_data, fh)
