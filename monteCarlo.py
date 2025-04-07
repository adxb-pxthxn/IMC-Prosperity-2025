import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def gather_stock_data():
    # file location: ./backtester/resources/round0/prices_round_-3.csv
    #                ./backtester/resources/round0/prices_round_-4.csv 




    # returns data['close']
    pass

def run():
    closing_prices = gather_stock_data()
    timeframe = 1 # This is an arbitrary number but should be changed to the number of the final time frame of the stock. 
    # Normally this is represented by number of years but should be shifted to however many "years" there are in the timeframe given
    future_trading_frame = 15
    num_simulations = 100

    newest = closing_prices[-1]
    oldest = closing_prices[0]

    total_growth = newest/oldest

    cagr = total_growth ** (1 / timeframe) - 1

    std_dev = pd.DataFrame(closing_prices).pct_change().std()
    stdDev = std_dev * math.sqrt(future_trading_frame)

    mu = cagr / future_trading_frame
    sigma = stdDev / math.sqrt(future_trading_frame)

    for i in range(num_simulations):
        dailyReturnPercent = np.random.normal(mu, sigma, future_trading_frame) + 1

        price_series = []
        price_series.append(closing_prices[-1])

        for j in dailyReturnPercent:
            price_series.append(price_series[-1] * j)
        
        plt.plot(price_series)
    plt.show()



    