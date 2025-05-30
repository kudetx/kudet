import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
def calculate_rsi(series, period=14):
    """
    Calculates the Relative Strength Index (RSI) for a given price series.
    :param series: ime series of closing prices.
    :param period: Look-back period for RSI calculation (default is 14).
    :return:
         pd.Series: RSI values ranging between 0 and 100.
    """

    #calculate daily price change
    delta = series.diff()
    #Just take the gains (positive value) (change sign)
    gain = (delta.where(delta > 0, 0)).rolling(window = period).mean()
    # Just take the losses (negative value)
    loss = (-delta.where(delta < 0, 0)).rolling(window = period).mean()
    #calculate RS --> avarage gain / avarage loss
    rs = gain / loss
    #RSI formula
    rsi = 100 - (100 / (1+rs))
    return rsi

def add_indicators(df, fundamentals=None):
    """
    Adds multiple technical indicators to the input DataFrame for use in machine learning models.
    Indicators include:
    - Moving Averages (MA5, MA20)
    - Relative Strength Index (RSI)
    - Momentum (10-day)
    - Bollinger Bands (20-day, 2 std dev)
    - MACD and MACD Signal (12, 26, 9 EMA)

    :param df: pd.DataFrame with 'Close' and 'Volume' columns.
    :param fundamentals: dict (optional), not used here but preserved for compatibility.
    :return: pd.DataFrame with added technical indicator columns.
    """

    # 5-day and 20-day Simple Moving Averages (trend detection)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()

    # 14-day RSI (momentum oscillator)
    df['RSI'] = calculate_rsi(df['Close'], period=14)

    # Momentum (price difference from 10 days ago)
    df['Momentum'] = df['Close'] - df['Close'].shift(10)

    # Bollinger Bands (volatility bands)
    df['Bollinger_Mid'] = df['Close'].rolling(window=20).mean()
    df['Bollinger_Std'] = df['Close'].rolling(window=20).std()
    df['Bollinger_Upper'] = df['Bollinger_Mid'] + 2 * df['Bollinger_Std']
    df['Bollinger_Lower'] = df['Bollinger_Mid'] - 2 * df['Bollinger_Std']

    # MACD (12-26 EMA difference) and Signal Line (9 EMA of MACD)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Drop any rows with missing values due to rolling calculations
    df.dropna(inplace=True)

    return df



def clean_data(df):
    """
     Removes rows with NaN values from the DataFrame.
    Typically used after adding technical indicators which introduce NaNs.
    :param df: Data Frame to be cleaned
    :return:
        pd.DataFrame: Cleaned DataFrame without NaN values.
    """
    return df.dropna()


def prepare_property(df, fundamentals):
    """
    Prepares and scales both technical and fundamental indicators for LSTM input.

    Steps:
    - Adds all supported technical indicators to the price DataFrame
    - Cleans the data by removing NaN rows (due to rolling/EMA ops)
    - Selects a feature set including both momentum and trend-based indicators
    - Scales the closing price for inverse transformation (for prediction outputs)
    - Scales fundamental metrics to match LSTM-compatible shape

    :param df: Raw OHLCV DataFrame from Yahoo Finance (must include 'Close', 'Volume')
    :param fundamentals: dict containing fundamental features (PE, PB, Net Income, etc.)
    :return:
        Tuple:
            - pd.DataFrame: Feature-engineered and scaled technical indicators
            - np.ndarray: Scaled fundamental features reshaped for LSTM input
            - MinMaxScaler: Scaler for 'Close' column to inverse-transform predictions
    """
    if df is None or df.empty:
        raise ValueError("None Dataset. Please enter valid symbol or check your connection")
    if not pd.api.types.is_numeric_dtype(df['Close']):
        df['Close'] = pd.to_numeric(df['Close'], errors= 'coerce')

    # Step 1: Add all technical indicators
    df = add_indicators(df, fundamentals)

    # Step 2: Drop rows with NaNs introduced by rolling calculations
    df = clean_data(df)

    # Step 3: Select relevant columns as features for LSTM input
    property_cols = [
        'Close', 'Volume', 'MA5', 'MA20', 'RSI',
        'Momentum', 'Bollinger_Upper', 'Bollinger_Lower',
        'MACD', 'MACD_Signal'
    ]
    df = df[property_cols]

    # Step 4: Scale target column ('Close') separately for inverse transform later
    scaler_y = MinMaxScaler()
    _ = scaler_y.fit_transform(df[['Close']])  # Fit only (used in inverse transform later)

    # Step 5: Normalize fundamental data
    fundamental_vec = np.array([list(fundamentals.values())])
    fund_scaler = MinMaxScaler()
    fundamental_scaled = fund_scaler.fit_transform(fundamental_vec)[0].reshape(1, -1)

    return df, fundamental_scaled, scaler_y
