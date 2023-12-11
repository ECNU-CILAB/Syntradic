import pandas as pd


class MovingAveragePriceCalculator:
    def __init__(self, window_size=10):
        self.window_size = window_size

    def __call__(self, data):
        price_series = data["open_price"]
        results = []
        sum_price, num = 0, 0
        for i, price in enumerate(price_series):
            sum_price += price
            num += 1
            if num > self.window_size:
                sum_price -= price_series[i - self.window_size]
                num -= 1
            results.append(sum_price / num)
        results = pd.Series(results)
        return results
