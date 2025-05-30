import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping  # <-- EN ÜSTE EKLE

def create_dataset(data, size, forecast):
    """
    Splits time series data into input-output sequences for LSTM training.
    :param data (np.ndarray): Scaled feature matrix.
    :param size (int): Number of time steps (window size) for each input sequence.
    :param forecast (int): Number of future steps to predict.
    :return:
        Tuple:
            - np.ndarray: Input features shaped as (samples, size, features).
            - np.ndarray: Corresponding target values shaped as (samples, forecast).
    """
    X, y = [], []
    for i in range(len(data) - size- forecast):
        X.append(data[i:i+size])
        y.append(data[i+ size:i + size+ forecast, 0 ])
    return np.array(X), np.array(y)

def train_lstm_lstm(df, fundamentals_vec_scaled, size=30, forecast=3, epochs=50):
    """
    Builds and trains a stacked LSTM model using technical and fundamental features.
    Uses all columns from the input DataFrame `df` (including Close, Volume, MA5, MA20, RSI, Momentum, Bollinger Bands, MACD).
    Prediction: only closing price (y is based on df['Close']).


    :param df: DataFrame with selected technical features.
    :param fundamentals_vec_scaled: Scaled fundamental data reshaped for LSTM.
    :param size: Number of time steps for input sequences (default is 60).
    :param forecast: Number of future steps to predict (default is 5).
    :param epochs: Number of training epochs (default is 50).
    :return:
        Tuple:
            - keras.Model: Trained LSTM model.
            - np.ndarray: Test input features.
            - np.ndarray: Test target values.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)
#Create input, output sequences
    X,y = create_dataset(scaled_data, size, forecast)
    if len(X) == 0 or len(y) ==0:
        raise ValueError("Eğitim için yeterli veri yok, daha kısa bir forecast değeri belirleyin ya da başka bir hisse kodu giriniz...")
#Repeat fundamental ector to match LSTM input shape
    fund_array = np.tile(fundamentals_vec_scaled, (X.shape[0], X.shape[1], 1))
    X_combined = np.concatenate([X, fund_array], axis=2)
#Split into train and test sets
    split = int(len(X) * 0.8)
    X_train, X_test = X_combined[:split], X_combined[split:]
    y_train, y_test = y[:split], y[split:]

#build the LSTM model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True,
                   input_shape = (X_combined.shape[1], X_combined.shape[2])))
    model.add(Dropout(0.2))  #%20 dropout
    model.add(LSTM(32))
    model.add(Dropout(0.2))  #2. dropout
    model.add(Dense(forecast))
    model.compile(optimizer='adam', loss='mse')

    # EarlyStopping callback’i tanımla
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    #train the model
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=32,
              verbose=1,
              validation_split=0.2,  # Eğitim verilerinin %20'si doğrulama için
              callbacks = [early_stop])
    return model, X_test, y_test
