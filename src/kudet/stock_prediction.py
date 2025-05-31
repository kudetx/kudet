# ───── Standart Kütüphaneler ─────
import os
import re
import sys
import math
import time
import unicodedata
# ───── Üçüncü Parti Kütüphaneler ─────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.layers import Dropout  # <-- Dropout eklendi

# ───── Yerel Modüller (Senin dosyaların) ─────
from indicators import add_indicators, prepare_property
from lstm_lstm import train_multi_input_model
from cache_utils import load_cached_data, save_to_cache, rate_limit_sleep, \
                         load_fundamentals_from_cache, save_fundamentals_to_cache, safe_download
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def format_symbol(symbol):
    """
    Formats user input into a standardized BIST stock symbol.
    :param symbol (str) :Raw input symbol from the user.
    :return:
      str: Cleaned and uppercase symbol ending with '.IS'.
    """
    # Sadece nokta içeren girişler için özel durum (".", "..", "..." vb.)
    if all(char == '.' for char in symbol):
        return symbol + '.IS'

    # Normal işlem
    symbol = unicodedata.normalize('NFKD', symbol).encode('ascii', 'ignore').decode('ascii')
    symbol = re.sub(r'[^A-Za-z0-9.]', '', symbol)
    symbol = symbol.upper()

    if symbol.endswith('.IS'):
        return symbol
    elif symbol.endswith('.'):
        return symbol  # Noktayı koru
    else:
        return symbol + '.IS'

def get_fundamentals(symbol):
    cached = load_fundamentals_from_cache(symbol)
    if cached:
        print(f"Temel veriler cacheden yüklendi: {symbol}")
        return cached
    try:
        time.sleep(3)  # Rate limit koruması
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if not info or 'forwardPE' not in info:
            raise ValueError("Temel veriler eksik veya API yanıtı boş")

        result = {
            'pe_ratio': info.get('forwardPE', 0),
            'pb_ratio': info.get('priceToBook', 0),
            'net_income': info.get('netIncomeToCommon', 0),
            'market_cap': info.get('marketCap', 0),
            'shares_outstanding': info.get('shareOutstanding', 0),
            'total_cash': info.get('totalCash', 0)
        }
        save_fundamentals_to_cache(symbol, result)
        return result
    except Exception as e:
        print(f"Temel veriler alınamadı: {e}")
        return {
            'pe_ratio': 0, 'pb_ratio': 0, 'net_income': 0,
            'market_cap': 0, 'shares_outstanding': 0, 'total_cash': 0
        }


def main():
    while True:
        try:
            inp_bist = input("BIST kodu giriniz (örnek: XU100.IS) veya çıkış için q: ").strip()
            if inp_bist.lower() == "q":
                confirm = input("Çıkış için 'q', devam için herhangi bir tuş: ").strip().lower()
                if confirm == "q":
                    print("Program sonlandırıldı...")
                    sys.exit(0)
                print("Program devam ediyor...\n")
                continue

            symbol = format_symbol(inp_bist)
            if symbol == '.IS':
                print("Geçersiz giriş, lütfen geçerli bir BIST kodu girin.\n")
                continue

            # Önce cache'den dene
            cached = load_cached_data(symbol, period='1y')

            if cached is not None:
                print(f"📁 Cache'den yüklendi: {symbol}")
                data = cached
            else:
                rate_limit_sleep(2)  # API sınırına takılmamak için bekleme
                try:
                    data = safe_download(symbol, period='1y', interval='1d')

                    if data is None or data.empty:
                        print("⚠️ Veri çekilemedi. Bu bir geçici bağlantı sorunu olabilir. "
                              "Lütfen biraz bekleyip tekrar deneyin.\n"
                              "Eğer bu hisse kodunun doğru olduğunu biliyorsanız, büyük ihtimalle geçici olarak API engeli uygulanmıştır.")
                        continue

                    try:
                        data = data.astype(float)
                    except Exception as e:
                        print(f"Veri türü dönüştürülürken hata oluştu: {e}")
                        continue

                    save_to_cache(symbol, data, period='1y')
                    print(f"✅ {symbol} yüklendi ve cache'e kaydedildi.")

                except Exception as e:
                    print(f"Veri çekme hatası: {e}")
                    continue

            print(f"\n{symbol} yüklendi")

            # Temel analiz verilerini al
            fundamentals = get_fundamentals(symbol)
            fundamentals_vec = np.array([list(fundamentals.values())])

            # LSTM parametreleri
            size = 45
            forecast = 20
            epochs = 60
            print(f"\n{symbol} için model eğitiliyor...\n")

            # Teknik göstergeler (MA, RSI, Momentum, MACD, Bollinger vb.) + temel oranlar içeren verileri hazırla
            processed, fundamental_scaled, scaler_y = prepare_property(data.copy(), fundamentals)

            # LSTM modelini eğit
            model, X_test_input, y_test = train_multi_input_model(
                processed,
                fundamental_scaled,
                size=size,
                forecast=forecast,
                epochs=epochs
            )

            # 1. Giriş verileri hazırlanıyor
            latest_input = processed[-size:].values
            X_input = latest_input.reshape((1, size, latest_input.shape[1])).astype('float32')

            # 2. Temel veriler float32 olarak garantileniyor
            fundamental_scaled = np.array(fundamental_scaled, dtype='float32')

            # Tahmin yap
            # Not: X_input ve fundamental_scaled ayrı verilecek
            pred_scaled = model.predict([X_input, fundamental_scaled])

            if pred_scaled is None or len(pred_scaled) == 0:
                raise ValueError("Tahmin başarısız oldu, model çıktı döndürmedi.")

            predicted_prices = scaler_y.inverse_transform(np.array(pred_scaled).reshape(-1, 1)).flatten()

            real_scaled = y_test[-1]
            real_prices = scaler_y.inverse_transform(real_scaled.reshape(-1, 1)).flatten()

            # Sonuçları yazdır
            print("\nTahmin Sonuçları:")
            for i, p in enumerate(predicted_prices):
                fark = ((p - real_prices[i]) / real_prices[i]) * 100
                print(f"+{i + 1} gün: Tahmin= {p:.2f} TL | Gerçek= {real_prices[i]:.2f} TL | Fark= {fark:.2f}%")

            # Model performansını hesapla
            rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))
            mae = mean_absolute_error(real_prices, predicted_prices)
            mape = np.mean(np.abs((real_prices - predicted_prices) / real_prices)) * 100
            r2 = r2_score(real_prices, predicted_prices)
            ortalama_sapma = np.mean([(abs(p - r) / r) * 100 for p, r in zip(predicted_prices, real_prices)])

            print("\n📊 Model Performansı:")
            print(f"🔹 R² skoru       : {r2:.4f}  → ", end="")
            if r2 > 0.9:
                print("Harika")
            elif r2 > 0.7:
                print("İyi")
            elif r2 > 0.3:
                print("Zayıf")
            else:
                print("Yetersiz")

            print(f"🔹 RMSE           : {rmse:.2f} TL")
            print(f"🔹 MAE            : {mae:.2f} TL")
            print(f"🔹 MAPE           : %{mape:.2f}")
            print(f"🔹 Ortalama Sapma : %{ortalama_sapma:.2f} → ", end="")
            if ortalama_sapma < 10:
                print("🎯 Mükemmel tahmin!")
            elif ortalama_sapma < 20:
                print("✅ Başarılı tahmin.")
            elif ortalama_sapma < 50:
                print("⚠️ Geliştirilebilir.")
            else:
                print("❌ Güvenilmez tahmin.")

            print("\nℹ️ Açıklama:")
            print("R² skoru 1'e ne kadar yakınsa, modelin tahmin gücü o kadar yüksek demektir.")
            print("Ortalama sapma ise tahminin, gerçek değerden ortalama ne kadar saptığını gösterir.")
            print("Hedefimiz: Ortalama sapma <%15 ve R² > 0.7 olmalı.")
            # Grafik çiz
            plt.figure(figsize=(12, 6))
            plt.plot(range(1, forecast + 1), predicted_prices,
                     label="Tahmin", linestyle='--', marker='x', color='red')
            plt.plot(range(1, forecast + 1), real_prices,
                     label="Gerçek", linestyle='-', marker='o', color='blue')

            plt.xlabel("Gün")
            plt.ylabel("Fiyat (TL)")
            plt.title(f"{symbol} - {forecast} Günlük LSTM Tahmin ve Gerçek Fiyatlar")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        except KeyboardInterrupt:
            print("\nProgram güvenli bir şekilde kapatılıyor...")
            plt.close('all')
            sys.exit(0)
        except Exception as e:
            print(f"\nBeklenmeyen hata: {e}")
            continue


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram sonlandırıldı.")
        plt.close('all')  # Açık grafikleri kapat
        sys.exit(0)
    except Exception as e:
        print(f"Program hatası: {e}")
        sys.exit(1)


