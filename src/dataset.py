import os
import logging
import numpy as np
from pathlib import Path
from features import extract_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(data_dir):
    data_dir = Path(data_dir).resolve()
    X = []
    y = []
    label_map = {}

    current_label = 0

    for genre in os.listdir(data_dir):
        genre_path = os.path.join(data_dir, genre)
        if not os.path.isdir(genre_path):
            continue

        label_map[genre] = current_label

        for file in os.listdir(genre_path):
            if file.endswith(".wav"):
                file_path = os.path.join(genre_path, file)
                print(f"İşleniyor: {file_path}")

                try:
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(current_label)
                except Exception as e:
                    logger.warning(f"Atlandı: {file_path} — Sebep: {e}")

        current_label += 1

    return np.array(X), np.array(y), label_map


if __name__ == "__main__":
    X, y, label_map = load_dataset("data")
    print("Toplam sample sayısı:", len(X))
    print("Feature boyutu:", X.shape)
    print("Label map:", label_map)