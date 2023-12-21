import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def get_rectangle_bounds_with_gap(timesteps, df, gap=0.1):
    min_price = df.loc[timesteps, "Low"].min()
    max_price = df.loc[timesteps, "High"].max()
    start_time = mdates.date2num(df["Time"][timesteps[0]])
    end_time = mdates.date2num(df["Time"][timesteps[-1]])
    # Adding a gap to the bounds
    price_gap = (max_price - min_price) * gap
    time_gap = (end_time - start_time) * gap
    return (
        start_time - time_gap,
        end_time + time_gap,
        min_price - price_gap,
        max_price + price_gap,
    )


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


class CandlestickGenerator:
    def __init__(self, patterns):
        self.patterns = patterns
        self.pattern_timesteps = []  # To store the timesteps where patterns occur
        self.trend_timesteps = (
            []
        )  # To store the timesteps of the trend following the pattern

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
                self.pattern_timesteps.append(i)  # Save pattern start timestep
                i += len(pattern_candles)
                last_candle = pattern_candles[-1]
                trend_candles = pattern.generate_trend(last_candle)
                candles.extend(trend_candles)
                self.trend_timesteps.append(
                    range(i, i + len(trend_candles))
                )  # Save trend timesteps
                i += len(trend_candles)
            else:
                last_close = candles[-1][-1] if candles else None
                candle = self.generate_random_candle(last_close)
                candles.append(candle)
                i += 1
        return candles


# Re-implementing the updated code provided by the user to generate the plot with rectangles


def improved_doji_pattern(last_close):
    """Generates a more realistic Doji pattern with a relative change."""
    # Small relative change for the open and close prices
    open_close_diff = last_close * np.random.uniform(-0.0001, 0.0001)
    open_price = last_close + open_close_diff
    close_price = last_close - open_close_diff
    high = max(open_price, close_price) + np.random.uniform(0.5, 3)
    low = min(open_price, close_price) - np.random.uniform(0.5, 3)
    return [(open_price, high, low, close_price)]


def refined_trend(candle, bars=5, is_bullish=None):
    """Generates a trend with a proper initial breakout and periods of indecision."""
    _, doji_high, doji_low, _ = candle

    if is_bullish is None:
        is_bullish = np.random.choice([True, False])

    trend_candles = []
    last_close = (
        doji_high if is_bullish else doji_low
    )  # Set initial last_close based on breakout direction

    # Generate the first candle with a proper breakout
    if is_bullish:
        open_price = np.random.uniform(
            doji_low, doji_high
        )  # Open can be anywhere within the Doji's range
        close_price = doji_high + np.random.uniform(
            1, 5
        )  # Close is above Doji's high for a bullish breakout
        high = max(close_price, doji_high) + np.random.uniform(0, 2)
        low = open_price - np.random.uniform(0, 2)
    else:
        open_price = np.random.uniform(
            doji_low, doji_high
        )  # Open can be anywhere within the Doji's range
        close_price = doji_low - np.random.uniform(
            1, 5
        )  # Close is below Doji's low for a bearish breakout
        high = open_price + np.random.uniform(0, 2)
        low = min(close_price, doji_low) - np.random.uniform(0, 2)

    trend_candles.append((open_price, high, low, close_price))
    last_close = close_price

    # Generate subsequent candles with variability and overall trend
    for _ in range(1, bars):
        # Decide if the candle is a trend continuation, indecision, or a slight retracement
        candle_type = np.random.choice(
            ["continuation", "indecision", "retracement"], p=[0.5, 0.4, 0.1]
        )
        if candle_type == "continuation":
            open_price = last_close
            close_price = open_price + np.random.uniform(1, 5) * (
                1 if is_bullish else -1
            )
        elif candle_type == "indecision":
            # Indecision candles have a small body
            open_price = last_close
            close_price = open_price + np.random.uniform(-1, 1)
        else:  # retracement
            open_price = last_close
            # Retracement candles move slightly against the trend
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


def hammer_trend():
    return lambda candle, bars=5: refined_trend(candle, bars, is_bullish=True)


# Initialize the Doji pattern
doji = CandlestickPattern("Doji", improved_doji_pattern, refined_trend)
hammer = CandlestickPattern("Hammer", hammer_pattern, hammer_trend())
inverted_hammer = CandlestickPattern(
    "Inverted Hammer", inverted_hammer_pattern, hammer_trend()
)

# Update the generator with the new patterns
# generator = CandlestickGenerator([doji, hammer, inverted_hammer])
generator = CandlestickGenerator([doji])
generated_data = generator.generate_data(100)

# Convert to DataFrame for plotting
df_candles = pd.DataFrame(generated_data, columns=["Open", "High", "Low", "Close"])
df_candles["Time"] = pd.to_datetime(df_candles.index, unit="D")

# Plotting the candlestick chart
fig, ax = plt.subplots(figsize=(20, 6))

# Plotting each candlestick individually
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

# Highlighting pattern and trend candles with a rectangle
for i, pattern_idx in enumerate(generator.pattern_timesteps):
    trend_idxs = list(generator.trend_timesteps[i])
    timesteps = [pattern_idx] + trend_idxs
    start_time, end_time, min_price, max_price = get_rectangle_bounds_with_gap(
        timesteps, df_candles
    )

    # Draw the rectangle
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

# Formatting the plot
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45)
plt.title("Candlestick Chart with Highlighted Patterns")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.savefig("test.png")
