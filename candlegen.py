import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def get_rectangle_bounds_with_gap(timesteps, df, gap=0.05):
    min_price = df.loc[timesteps, "Low"].min()
    max_price = df.loc[timesteps, "High"].max()
    start_time = mdates.date2num(df["Time"][timesteps[0]])
    end_time = mdates.date2num(df["Time"][timesteps[-1]])

    price_gap = (max_price - min_price) * gap
    time_gap = (end_time - start_time) * gap
    return (
        start_time - time_gap,
        end_time + time_gap,
        min_price - price_gap,
        max_price + price_gap,
    )


class CandlestickPattern:
    """Class to represent a candlestick pattern and its subsequent pre and post trends."""

    def __init__(
        self, name, pattern_function, pre_trend_function=None, post_trend_function=None
    ):
        self.name = name
        self.pattern_function = pattern_function
        self.pre_trend_function = pre_trend_function
        self.post_trend_function = post_trend_function

    def generate_pre_trend(self, last_candle, bars=5):
        """Generates the pre-trend before the pattern."""
        return (
            self.pre_trend_function(last_candle, bars)
            if self.pre_trend_function
            else []
        )

    def generate_pattern(self, last_close):
        """Generates the candlestick pattern using the last close value."""
        return self.pattern_function(last_close)

    def generate_post_trend(self, last_candle, bars=5):
        """Generates the post-trend following the pattern."""
        return (
            self.post_trend_function(last_candle, bars)
            if self.post_trend_function
            else []
        )


class CandlestickGenerator:
    def __init__(self, patterns):
        self.patterns = patterns
        self.pre_trend_timesteps = []
        self.pattern_timesteps = []
        self.post_trend_timesteps = []

    def generate_data(self, num_candles, pattern_chance=0.2):
        candles = []
        i = 0
        while i < num_candles:
            last_candle = candles[-1] if candles else None

            # Don't ever generate a pattern candle for the first candle
            if last_candle and random.random() < pattern_chance and i > 0:
                pattern = random.choice(self.patterns)

                last_candle = candles[-1]
                pre_trend_candles = pattern.generate_pre_trend(last_candle)
                candles.extend(pre_trend_candles)
                self.pre_trend_timesteps.append(range(i, i + len(pre_trend_candles)))
                i += len(pre_trend_candles)

                last_candle = candles[-1]
                pattern_candles = pattern.generate_pattern(last_candle[-1])
                candles.extend(pattern_candles)
                self.pattern_timesteps.append(i)
                i += len(pattern_candles)

                last_candle = candles[-1]
                post_trend_candles = pattern.generate_post_trend(last_candle)
                candles.extend(post_trend_candles)
                self.post_trend_timesteps.append(range(i, i + len(post_trend_candles)))
                i += len(post_trend_candles)
            else:
                last_close = candles[-1][-1] if candles else None
                candle = self.generate_random_candle(last_close)
                candles.append(candle)
                i += 1
        return candles

    def generate_random_candle(self, last_close=None):
        if last_close is None:
            last_close = np.random.uniform(100, 200)
        open_price = np.random.uniform(last_close - 5, last_close + 5)
        close_price = np.random.uniform(open_price - 10, open_price + 10)
        high = max(open_price, close_price) + np.random.uniform(0, 5)
        low = min(open_price, close_price) - np.random.uniform(0, 5)
        return open_price, high, low, close_price


def improved_doji_pattern(last_close):
    """Generates a more realistic Doji pattern with a relative change."""

    open_close_diff = last_close * np.random.uniform(-0.0001, 0.0001)
    open_price = last_close + open_close_diff
    close_price = last_close - open_close_diff
    high = max(open_price, close_price) + np.random.uniform(0.5, 3)
    low = min(open_price, close_price) - np.random.uniform(0.5, 3)
    return [(open_price, high, low, close_price)]


def refined_trend(
    candle, bars=5, breakout=True, is_bullish=None, probabilities=[0.5, 0.4, 0.1]
):
    _, prior_high, prior_low, _ = candle

    if is_bullish is None:
        is_bullish = np.random.choice([True, False])

    trend_candles = []
    last_close = prior_high if is_bullish else prior_low

    baseline = np.random.uniform(prior_low, prior_high)
    if is_bullish:
        open_price = np.random.uniform(prior_low, prior_high)
        if breakout:
            baseline = prior_high
        close_price = baseline + np.random.uniform(1, 5)
        high = max(close_price, prior_high) + np.random.uniform(0, 2)
        low = open_price - np.random.uniform(0, 2)
    else:
        open_price = np.random.uniform(prior_low, prior_high)
        if breakout:
            baseline = prior_low
        close_price = baseline - np.random.uniform(1, 5)
        high = open_price + np.random.uniform(0, 2)
        low = min(close_price, prior_low) - np.random.uniform(0, 2)

    trend_candles.append((open_price, high, low, close_price))
    last_close = close_price

    for _ in range(1, bars):
        candle_type = np.random.choice(
            ["continuation", "indecision", "retracement"], p=probabilities
        )
        if candle_type == "continuation":
            open_price = last_close
            close_price = open_price + np.random.uniform(1, 5) * (
                1 if is_bullish else -1
            )
        elif candle_type == "indecision":
            open_price = last_close
            close_price = open_price + np.random.uniform(-1, 1)
        else:
            open_price = last_close

            close_price = open_price - np.random.uniform(0, 2) * (
                1 if is_bullish else -1
            )

        high = max(open_price, close_price) + np.random.uniform(0, 2)
        low = min(open_price, close_price) - np.random.uniform(0, 2)
        trend_candles.append((open_price, high, low, close_price))
        last_close = close_price

    return trend_candles


def hammer_pattern(last_close):
    """Generates a Hammer candlestick pattern."""
    body_size = np.random.uniform(0.5, 1.5)
    lower_shadow = body_size * np.random.uniform(2, 3)
    upper_shadow = np.random.uniform(0, 0.5)

    close_price = last_close - body_size
    open_price = close_price + body_size
    low = close_price - lower_shadow
    high = max(open_price, close_price) + upper_shadow

    return [(open_price, high, low, close_price)]


def inverted_hammer_pattern(last_close):
    """Generates an Inverted Hammer candlestick pattern."""
    body_size = np.random.uniform(0.5, 1.5)
    upper_shadow = body_size * np.random.uniform(2, 3)
    lower_shadow = np.random.uniform(0, 0.5)

    open_price = last_close - body_size
    close_price = open_price + body_size
    high = close_price + upper_shadow
    low = min(open_price, close_price) - lower_shadow

    return [(open_price, high, low, close_price)]


def hammer_pre_trend():
    return lambda candle, bars=5: refined_trend(
        candle, bars, is_bullish=False, probabilities=[0.7, 0.2, 0.1]
    )


def hammer_post_trend():
    return lambda candle, bars=5: refined_trend(
        candle, bars, is_bullish=True, probabilities=[0.7, 0.2, 0.1]
    )


doji = CandlestickPattern("Doji", improved_doji_pattern, None, refined_trend)
hammer = CandlestickPattern(
    "Hammer", hammer_pattern, hammer_pre_trend(), hammer_post_trend()
)
inverted_hammer = CandlestickPattern(
    "Inverted Hammer", inverted_hammer_pattern, hammer_pre_trend(), hammer_post_trend()
)
# generator = CandlestickGenerator([doji])
# generator = CandlestickGenerator([hammer])
# generator = CandlestickGenerator([doji, hammer])
# generator = CandlestickGenerator([inverted_hammer])
generator = CandlestickGenerator([doji, hammer, inverted_hammer])
generated_data = generator.generate_data(100, pattern_chance=0.1)


df_candles = pd.DataFrame(generated_data, columns=["Open", "High", "Low", "Close"])
df_candles["Time"] = pd.to_datetime(df_candles.index, unit="D")


fig, ax = plt.subplots(figsize=(20, 6))


for i in range(len(df_candles)):
    color = "green" if df_candles["Close"][i] > df_candles["Open"][i] else "red"
    ax.plot(
        [df_candles["Time"][i], df_candles["Time"][i]],
        [df_candles["Low"][i], df_candles["High"][i]],
        color="black",
    )
    ax.plot(
        [df_candles["Time"][i], df_candles["Time"][i]],
        [df_candles["Open"][i], df_candles["Close"][i]],
        color=color,
        linewidth=5,
    )


for i, pattern_idx in enumerate(generator.pattern_timesteps):
    pre_trend_idxs = (
        list(generator.pre_trend_timesteps[i]) if generator.pre_trend_timesteps else []
    )
    post_trend_idxs = list(generator.post_trend_timesteps[i])

    # Combining pre-trend, pattern, and post-trend time steps
    timesteps = pre_trend_idxs + [pattern_idx] + post_trend_idxs

    start_time, end_time, min_price, max_price = get_rectangle_bounds_with_gap(
        timesteps, df_candles
    )

    width = end_time - start_time
    height = max_price - min_price
    rect = plt.Rectangle(
        (start_time, min_price),
        width,
        height,
        linewidth=1,
        edgecolor="blue",
        facecolor="none",
    )
    ax.add_patch(rect)


ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45)
plt.title("Candlestick Chart with Highlighted Patterns")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.savefig("test.png")
