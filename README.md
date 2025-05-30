# kudet project
# KUDET — Kısa Vadeli Borsa Tahmin Sistemi (BIST Hisse Fiyat Tahmin Sistemi--> LSTM + Teknik & Temel Analiz)

**KUDET** (Kısa Vadeli Uzunlukta Derin Öğrenme Tahminleyicisi), Borsa İstanbul (BIST)hisse senedi verilerini kullanarak,
kısa vadeli kapanış fiyatı tahminleri sunan bir yapay zeka projesidir. Teknik analiz ve temel analiz verilerini 
birleştirerek, LSTM (Long Short-Term Memory) mimarisi ile tahmin yapar. Proje, Python ile yazılmış olup 
`yfinance` API’si üzerinden veri çekimi yapar ve makine öğrenmesi ile kısa vadeli fiyat tahminleri üretir. 
Finans mühendisliği, yatırım analizi, yapay zeka ve veri bilimi alanlarında kullanılabilir.

## Projenin Hedefi
KUDET'in amacı, yatırımcıların karar destek süreçlerinde kullanılabilecek bir fiyat tahmin altyapısı sağlamaktır. 
Bunu sağlamak için:

##  Kullanılan Yaklaşımlar
*  Teknik göstergeler: (RSI, MA5, MA20, Momentum, Bollinger Bands, MACD vb.) hesaplanır
*  Temel analiz verileri: (F/K, PD/DD, Net Income, Market Cap vb.) YF API ile alınır
*  Veri Önişleme: MinMaxScaler ile ölçekleme, eksik verilerin temizlenmesi
*  Model: İki katmanlı LSTM + Dropout + EarlyStopping ile overfitting önleme. Temel verilerle eğitilir
*  Test: PyTest ile birim testleri (%100 başarı)
*  Kapanış fiyatı tahmini ve grafiksel analiz gerçekleştirilir

---

##  Kullanılan Teknolojiler

| Alan             | Kütüphane                       |
| ---------------- | ------------------------------- |
| Veri Analizi     | pandas, numpy                   |
| Görselleştirme   | matplotlib                      |
| Finansal Veri    | yfinance                        |
| Makine Öğrenmesi | tensorflow, keras               |
| Ölçekleme & Skor | sklearn (MinMaxScaler, R², MSE) |
| Test             | pytest                          |

---

## Proje Klasör Yapısı

```
kudet/
├── src/
│   └── kudet/
│       ├── indicators.py         # Teknik göstergelerin hesaplandığı modül
│       ├── cache_utils.py        # Cache mekanizması ve veri önbellekleme ve tekrar kullanımı
│       ├── lstm_lstm.py          # LSTM mimarisi ve eğitimi
│       ├── stock_prediction.py   # Ana çalışma dosyası (interaktif terminal)
│       └── data_cache/           # İndirilen veriler ve cache (otomatik oluşur)
├── tests/
│   └── test_fonksiyonlar.py      # Test fonksiyonları (pytest ile çalıştırılır)
├── LICENSE                       # MIT Lisansı
├── README.md                     # Bu doküman
├── requirements.txt              # Gereken kütüphaneler
├── pyproject.toml                # Paket konfigürasyonu
```

---

## Kurulum Adımları

1. Sanal ortam oluştur
python3 -m venv venv

2. Sanal ortamı etkinleştir
# MacOS / Linux:
source venv/bin/activate
# Windows (CMD):
venv\Scripts\activate
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

3. Gereken kütüphaneleri yükle
pip install -r requirements.txt

4. Uygulamayı başlat
python src/kudet/stock_prediction.py


> İlk çalıştırmada `data_cache/` klasörü otomatik oluşur ve veriler cache'e alınır.

---

## Testler ve Doğrulama

Projede profesyonel test senaryoları yer almaktadır. 
Tüm kritik bileşenler tests/test_fonksiyonlar.py içinde test edilmektedir:

Testleri çalıştırmak için:
pytest tests/test_fonksiyonlar.py

***Test Kapsamı ve Amaçları***
Bu projede model güvenilirliğini ve veri işleme bütünlüğünü sağlamak amacıyla aşağıdaki test senaryoları 
gerçekleştirilmiştir:

1. RSI (Relative Strength Index) Hesaplama Doğruluğu

RSI fonksiyonunun ürettiği çıktının giriş serisiyle aynı uzunlukta olduğu doğrulanır.
Boş seri gönderildiğinde hata vermemesi ve NaN içeren mantıklı bir seri dönmesi beklenir.

2. Teknik Göstergelerin Eksiksiz Eklenmesi
add_indicators() fonksiyonu ile:
MA5, MA20
RSI
Momentum
Bollinger Üst/Alt Bantlar
MACD ve MACD_Signal gibi teknik göstergelerin DataFrame’e eksiksiz eklendiği test edilir.

3. Temel Analiz ve Teknik Göstergelerin LSTM Giriş Şekli

prepare_property() fonksiyonu:
Teknik ve temel analiz verilerini birlikte işler.
Girişlerin boyutu (shape) kontrol edilir.
Normalize edilmiş temel veri vektörünün LSTM giriş şekline uygunluğu doğrulanır.

4. LSTM Eğitiminin Başarıyla Gerçekleşmesi (Minimal Veri ile)

Örneklenmiş küçük veri setiyle model eğitimi başlatılır.
Eğitim sürecinde hata alınmaması ve modelin çıktı vermesi beklenir.

5. Cache Sistemi ve Bozuk Dosya Yedekleme Mekanizması

save_to_cache() ve load_cached_data() ile:
CSV dosyasının başarıyla kaydedilip okunabildiği test edilir.
Dosya bozuksa .bak uzantısıyla yedeklenip sistemin otomatik geri toparlama yaptığı kontrol edilir.

6. İstisnai Durumlar ile Baş Etme

Boş DataFrame, eksik Close verisi, hatalı format gibi senaryolar simüle edilir.
Bu durumlarda ValueError, TypeError gibi kontrollü istisnalar beklenir; sistem çökmemelidir.
Bu testler sayesinde, modelin sadece doğru veriyle değil, aynı zamanda bozuk veya eksik verilerle karşılaştığında da 
dayanıklı ve öngörülebilir şekilde davranması sağlanır.
---

## Model Mimarisi

* 2 Katmanlı LSTM:

  * İlk katman: `return_sequences=True`
  * Dropout: %20 oranında
  * Dense çıkış katmanı: Tahmin günü sayısına göre (`forecast=3`)
* Girdi Özellikleri:

  * Teknik göstergeler (10+ özellik)
  * Temel oranlar (6 özellik)
* Çıkış: Son 3 gün için kapanış fiyat tahmini

---

##  Özellikler

✅ YF API üzerinden canlı veri çekimi 

✅ Rate limit koruması (otomatik bekleme ve retry)

✅ Bozuk veri ve JSON dosyaları otomatik yedeklenir

✅ Tüm hesaplamalar cache'e alınır (hızlı çalıştırma)

✅ Pytest ile %100 modül test edilebilirliği

✅ Grafiksel analiz ve R² skoru görselleştirme

---

##  Sorumluluk Reddi

> Bu proje yatırım tavsiyesi içermez. Gerçek hisse yatırımı için profesyonel finansal danışmanlık alınmalıdır. 
> Bu sistem, eğitim ve araştırma amaçlıdır.

---

##  Lisans

Bu proje MIT Lisansı ile lisanslanmıştır. Ayrıntılar için `LICENSE` dosyasını inceleyiniz.

---

##  Katkı ve Geliştirme

Katkıda bulunmak isteyenler aşağıdaki yollarla katkı sağlayabilir:

* Yeni teknik göstergeler ekleyerek
* Model mimarisini iyileştirerek
* Grafiksel arayüz (GUI) önerileri sunarak
* Test kapsamını genişleterek

Pull Request'ler memnuniyetle karşılanır 

---

Teşekkürler 

> KUDET — Finansal Tahmin Platformu
