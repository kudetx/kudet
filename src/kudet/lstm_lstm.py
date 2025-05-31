import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model


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

def train_multi_input_model(df, fundamentals_vec_scaled, size=45, forecast=20, epochs=60):
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

    X, y = create_dataset(scaled_data, size, forecast)

    if len(X) == 0 or len(y) == 0:
        raise ValueError("Eğitim için yeterli veri yok.")

    # Teknik veri input
    input_teknik = Input(shape=(X.shape[1], X.shape[2]))
    x = LSTM(64, return_sequences=True)(input_teknik)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)

    # Temel veri input
    input_temel = Input(shape=(fundamentals_vec_scaled.shape[1],))
    f = Dense(32, activation='relu')(input_temel)
    f = Dropout(0.1)(f)

    # Birleştirme ve çıkış
    combined = Concatenate()([x, f])
    output = Dense(forecast)(combined)

    model = Model(inputs=[input_teknik, input_temel], outputs=output)
    model.compile(optimizer='adam', loss='mse')

    # Temel verileri tekrar et
    fund_array = np.repeat(fundamentals_vec_scaled, X.shape[0], axis=0)

    # Eğitim / test ayır
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    f_train, f_test = fund_array[:split], fund_array[split:]
    y_train, y_test = y[:split], y[split:]

    # Erken durdurma: val_loss 5 epoch boyunca iyileşmezse durur ve en iyi ağırlıkları geri yükler
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Modeli eğit
    model.fit(
        [X_train, f_train], y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],  # <-- erken durdurma eklendi
        verbose=1
    )

    # Eğitilmiş modeli ve test verilerini döndür
    return model, [X_test, f_test], y_test
