#  beat-ai

A music genre classification project that extracts audio features and predicts the genre using machine learning.

##  Features
- 41 audio features: MFCC, Chroma, Spectral Centroid, Zero Crossing Rate
- SVM algorithm with 71% accuracy
- 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

##  Project Structure
```
beat-ai/
├── src/
│   ├── features.py   # Feature extraction
│   ├── dataset.py    # Dataset loader
│   ├── train.py      # Model training
│   └── predict.py    # Prediction
├── tests/
│   └── test_beat_ai.py
└── requirements.txt
```

##  Installation
```bash
pip install -r requirements.txt
```

##  Dataset
Uses the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).
Download and place the genre folders inside the `data/` directory.

##  Usage
Train the model:
```bash
python src/train.py
```

Predict a genre:
```bash
python src/predict.py "audio_file.wav"
```

## 🧪 Tests
```bash
python -m pytest tests/ -v
```
