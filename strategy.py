import numpy as np
import pandas as pd


def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trend Following + RSI Confirmation Strategy

    Strategy Overview
    -----------------
    This strategy combines:
    1. Trend detection using EMA20 and EMA50.
    2. Momentum confirmation using RSI(14).

    Trading Rules
    -------------
    Long Position (1):
        - EMA20 > EMA50
        - RSI > 60

    Short Position (-1):
        - EMA20 < EMA50
        - RSI < 40

    Neutral Position (0):
        - All remaining cases

    Parameters
    ----------
    df : pd.DataFrame
        Historical OHLCV market data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - ema20
        - ema50
        - rsi
        - position
    """

    # Create a copy to avoid modifying original data
    df = df.copy()

    # ==========================================================
    # Price Series
    # ==========================================================
    close = df["Close"]

    # ==========================================================
    # Exponential Moving Averages
    # ==========================================================
    df["ema20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    df["ema50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    # ==========================================================
    # RSI(14)
    # ==========================================================
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(
        window=14,
        min_periods=14
    ).mean()

    avg_loss = loss.rolling(
        window=14,
        min_periods=14
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))

    # ==========================================================
    # Position Generation
    # ==========================================================
    df["position"] = 0

    long_condition = (
        (df["ema20"] > df["ema50"])
        & (df["rsi"] > 60)
    )

    short_condition = (
        (df["ema20"] < df["ema50"])
        & (df["rsi"] < 40)
    )

    df.loc[long_condition, "position"] = 1
    df.loc[short_condition, "position"] = -1

    # Ensure valid integer positions
    df["position"] = (
        df["position"]
        .fillna(0)
        .astype(int)
    )

    return df