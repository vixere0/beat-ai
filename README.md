#  beat-ai

Müzik türü sınıflandırma projesi. Ses dosyalarından özellik çıkararak makine öğrenmesi ile türü tahmin eder.

##  Özellikler
- MFCC, Chroma, Spectral Centroid, Zero Crossing Rate ile 41 özellik
- SVM algoritması ile %71 doğruluk
- 10 müzik türü: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

##  Proje Yapısı
```
beat-ai/
├── src/
│   ├── features.py   # Özellik çıkarımı
│   ├── dataset.py    # Veri yükleme
│   ├── train.py      # Model eğitimi
│   └── predict.py    # Tahmin
├── tests/
│   └── test_beat_ai.py
└── requirements.txt
```

##  Kurulum
```bash
pip install -r requirements.txt
```

##  Kullanım
Model eğitimi:
```bash
python src/train.py
```

Tahmin:
```bash
python src/predict.py "ses_dosyasi.wav"
```

##  Testler
```bash
python -m pytest tests/ -v
```

