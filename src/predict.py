import joblib
import sys
from features import extract_features

model = joblib.load(r"c:\Users\Abdullah Baran\OneDrive\Desktop\beat-ai\model\genre_classifier.pkl")
label_map = joblib.load(r"c:\Users\Abdullah Baran\OneDrive\Desktop\beat-ai\model\label_map.pkl")
scaler = joblib.load(r"c:\Users\Abdullah Baran\OneDrive\Desktop\beat-ai\model\scaler.pkl")

reverse_map = {v: k for k, v in label_map.items()}

audio_path = sys.argv[1]
features = extract_features(audio_path)
features = scaler.transform([features])
prediction = model.predict(features)[0]

print(f"Tahmin edilen tür: {reverse_map[prediction]}")