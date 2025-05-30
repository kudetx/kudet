import sys
import os
import pytest
import numpy as np
import pandas as pd

# src/kudet içindeki modülleri içe alabilmek için PYTHONPATH ayarı
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'kudet')))

# Yerel modülleri içe aktar
from indicators import calculate_rsi, add_indicators, prepare_property
from lstm_lstm import train_lstm_lstm
from cache_utils import save_to_cache, load_cached_data, backup_corrupted_file
from stock_prediction import format_symbol

# ─────────────── RSI ve Göstergeler ───────────────

def test_calculate_rsi_output_length():
    prices = pd.Series([i for i in range(30)])
    rsi = calculate_rsi(prices)
    assert len(rsi) == 30

def test_add_indicators_columns():
    df = pd.DataFrame({
        'Close': np.linspace(100, 200, 50),
        'Volume': np.random.randint(1000, 5000, 50)
    })
    enriched_df = add_indicators(df.copy())
    expected_columns = ['MA5', 'MA20', 'RSI', 'Momentum', 'Bollinger_Upper', 'Bollinger_Lower', 'MACD', 'MACD_Signal']
    for col in expected_columns:
        assert col in enriched_df.columns

# ─────────────── Veri Kalitesi ───────────────

def test_prepare_property_with_missing_values():
    df = pd.DataFrame({
        'Close': [100, 105, None, 110, 115],
        'Volume': [1000, 1200, 1300, None, 1400]
    })
    fundamentals = {
        'pe_ratio': 10, 'pb_ratio': 1.5, 'net_income': 500000,
        'market_cap': 200000000, 'shares_outstanding': 100000000,
        'total_cash': 15000000
    }
    with pytest.raises(ValueError):
        prepare_property(df.copy(), fundamentals)

# ─────────────── prepare_property temel kontrol ───────────────

def test_prepare_property_shapes():
    df = pd.DataFrame({
        'Close': np.linspace(100, 200, 60),
        'Volume': np.random.randint(1000, 5000, 60)
    })
    fundamentals = {
        'pe_ratio': 10, 'pb_ratio': 1.5, 'net_income': 500000,
        'market_cap': 200000000, 'shares_outstanding': 100000000,
        'total_cash': 15000000
    }
    X, f_vec, scaler_y = prepare_property(df.copy(), fundamentals)
    assert isinstance(X, pd.DataFrame)
    assert f_vec.shape == (1, 6)

# ─────────────── LSTM Eğitim Testi ───────────────

def test_lstm_training_minimal_input():
    df = pd.DataFrame({
        'Close': np.linspace(100, 120, 40),
        'Volume': np.random.randint(1000, 2000, 40)
    })
    fundamentals = {
        'pe_ratio': 15, 'pb_ratio': 2.1, 'net_income': 750000,
        'market_cap': 500000000, 'shares_outstanding': 150000000,
        'total_cash': 30000000
    }
    processed, f_vec, _ = prepare_property(df.copy(), fundamentals)
    model, X_test, y_test = train_lstm_lstm(processed, f_vec, size=10, forecast=2, epochs=1)
    assert X_test.shape[0] > 0 and y_test.shape[0] > 0

# ─────────────── Cache Testi ───────────────

def test_cache_save_and_load(tmp_path):
    symbol = "TEST.IS"
    df = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=5),
        'Close': [100, 101, 102, 103, 104]
    })
    df.set_index('Date', inplace=True)

    orig_dir = os.getcwd()
    os.chdir(tmp_path)

    try:
        save_to_cache(symbol, df)
        loaded_df = load_cached_data(symbol)
        assert loaded_df is not None
        assert 'Close' in loaded_df.columns
        assert len(loaded_df) == 5
    finally:
        os.chdir(orig_dir)

# ─────────────── Edge Case: Empty RSI ───────────────

def test_rsi_with_empty_series():
    empty = pd.Series([], dtype=float)
    rsi = calculate_rsi(empty)
    assert rsi.empty

# ─────────────── Cache Bozulduğunda Otomatik Silme ───────────────

def test_cache_auto_remove_on_error(tmp_path):
    symbol = "TESTERROR.IS"
    fake_cache_file = tmp_path / f"{symbol}_1y.csv"

    with open(fake_cache_file, 'w') as f:
        f.write("bozuk,veri")

    os.chdir(tmp_path)
    try:
        result = load_cached_data(symbol)
        assert result is None or result.empty
    finally:
        os.chdir(os.path.abspath(os.path.join(tmp_path, '..')))

# ─────────────── Entegrasyon Testi ───────────────
def test_main_integration_end_to_end():
    symbol = format_symbol("VESTL")
    data = load_cached_data(symbol)
    if data is None:
        pytest.skip("Cache verisi bulunamadı: VESTL.IS")

    fundamentals = {
        'pe_ratio': 10, 'pb_ratio': 2, 'net_income': 500000,
        'market_cap': 1e8, 'shares_outstanding': 5e6, 'total_cash': 1e7
    }

    processed, f_vec, scaler_y = prepare_property(data.copy(), fundamentals)
    model, X_test, y_test = train_lstm_lstm(processed, f_vec, size=20, forecast=5, epochs=1)
    y_pred = model.predict(X_test)
    assert y_pred is not None and len(y_pred) > 0
