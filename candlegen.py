import random
import numpy as np


class CandlestickPattern:
    """Class to represent a candlestick pattern and its subsequent trend."""

    def __init__(self, name, pattern_function, trend_function):
        self.name = name
        self.pattern_function = pattern_function
        self.trend_function = trend_function

    def generate_pattern(self, last_close):
        """Generates the candlestick pattern using the last close value."""
        return self.pattern_function(last_close)

    def generate_trend(self, last_candle):
        """Generates the trend following the pattern."""
        return self.trend_function(last_candle)


def improved_doji_pattern(last_close):
    """Generates an improved Doji pattern using the last close value."""
    # A Doji has a similar open and close price, close to the last close value
    open_close = np.random.uniform(last_close - 1, last_close + 1)
    high = open_close + np.random.uniform(1, 3)
    low = open_close - np.random.uniform(1, 3)
    return [(open_close, high, low, open_close)]


def doji_trend(last_close):
    """Generates an uptrend after a Doji pattern."""
    trend_candles = []
    for _ in range(3):  # Generate 3 candles for the trend
        open_price = np.random.uniform(last_close, last_close + 2)
        close_price = np.random.uniform(open_price, open_price + 5)
        high = close_price + np.random.uniform(0, 2)
        low = open_price - np.random.uniform(0, 2)
        trend_candles.append((open_price, high, low, close_price))
        last_close = close_price
    return trend_candles


# Update the Doji pattern function
doji = CandlestickPattern("Doji", improved_doji_pattern, doji_trend)


# Update the CandlestickGenerator class to pass last_close to the pattern function
class CandlestickGenerator:
    def __init__(self, patterns):
        self.patterns = patterns

    def generate_random_candle(self, last_close=None):
        if last_close is None:
            last_close = np.random.uniform(100, 200)
        open_price = np.random.uniform(last_close - 5, last_close + 5)
        close_price = np.random.uniform(open_price - 10, open_price + 10)
        high = max(open_price, close_price) + np.random.uniform(0, 5)
        low = min(open_price, close_price) - np.random.uniform(0, 5)
        return open_price, high, low, close_price

    def generate_data(self, num_candles, pattern_chance=0.2):
        candles = []
        i = 0
        while i < num_candles:
            if random.random() < pattern_chance and i > 0:
                pattern = random.choice(self.patterns)
                last_close = candles[-1][-1] if candles else None
                pattern_candles = pattern.generate_pattern(last_close)
                candles.extend(pattern_candles)
                i += len(pattern_candles)
                last_candle = pattern_candles[-1]
                trend_candles = pattern.generate_trend(last_candle[-1])
                candles.extend(trend_candles)
                i += len(trend_candles)
            else:
                last_close = candles[-1][-1] if candles else None
                candle = self.generate_random_candle(last_close)
                candles.append(candle)
                i += 1
        return candles


# Initialize the generator with the updated pattern
generator = CandlestickGenerator([doji])

# Generate new candlestick data with the improved Doji pattern
new_generated_data = generator.generate_data(20)

import ipdb

ipdb.set_trace()
