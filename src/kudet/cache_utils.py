import os
import time
import json
import pandas as pd
import yfinance as yf

DATA_FOLDER = "./data_cache"

def backup_corrupted_file(path):
    if os.path.exists(path):
        backup_path = path + ".bak"
        os.rename(path, backup_path)
        print(f"Bozuk dosya yedeklendi: {backup_path}")

def ensure_data_folder():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

def get_cached_filepath(symbol, period='1y'):
    return os.path.join(DATA_FOLDER, f"{symbol.replace('.IS','')}_{period}.csv")

def save_to_cache(symbol, df, period='1y'):
    ensure_data_folder()
    if df.empty:
        print(f"Uyarı: {symbol} verisi boş. Cache'e yazılmadı.")
        return
    file_path = get_cached_filepath(symbol, period)
    df = df.copy()

    if 'Date' not in df.columns:
        df.reset_index(inplace=True)

    if 'Date' not in df.columns:
        raise ValueError("DataFrame'de 'Date' sütunu bulunamadı, cache yazılamaz!")

    df.to_csv(file_path, index=False)

def load_cached_data(symbol, period='1y'):
    ensure_data_folder()
    file_path = get_cached_filepath(symbol, period)
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            return df
        except Exception as e:
            print(f"Hata: {file_path} dosyası okunamadı. {e}")
            print("🛉 Bozuk cache siliniyor ve veri yeniden indirilecek...\n")
            backup_corrupted_file(file_path)
            return None
    return None

def rate_limit_sleep(seconds=2):
    print(f"⏳ API yüklenmesini engellememek için {seconds} saniye bekleniyor...")
    time.sleep(seconds)

def get_fundamentals_path(symbol):
    ensure_data_folder()
    return os.path.join(DATA_FOLDER, f"{symbol.replace('.IS', '')}_fundamentals.json")

def save_fundamentals_to_cache(symbol, data):
    path = get_fundamentals_path(symbol)
    with open(path, 'w') as f:
        json.dump(data, f)

def load_fundamentals_from_cache(symbol):
    path = get_fundamentals_path(symbol)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Hata: {path} dosyası okunamadı. {e}")
            print("🛉 Bozuk fundamentals cache siliniyor, yeniden alınacak...\n")
            backup_corrupted_file(path)
            return None
    return None

def safe_download(symbol, period='1y', interval='1d', max_retries=3):
    """
    YF'den veri indirirken otomatik retry ve hata kontrolü uygular.
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Veri indiriliyor ({attempt}/{max_retries}): {symbol}")
            data = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
            if data is not None and not data.empty:
                return data
            else:
                print("Veri boş döndü.")
        except Exception as e:
            if 'rate limit' in str(e).lower() or 'too many requests' in str(e).lower():
                wait = 10 * attempt
                print(f"Rate limit'e takıldı, {wait} saniye bekleniyor...")
                time.sleep(wait)
            else:
                print(f"Veri çekme hatası: {e}")
        time.sleep(8)  # Her deneme arası küçük bekleme
    print("Veri indirilemedi, maksimum deneme hakkı aşıldı.")
    return None
