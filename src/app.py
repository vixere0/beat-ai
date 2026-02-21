import gradio as gr
import joblib
from features import extract_features

model = joblib.load(r"c:\Users\Abdullah Baran\OneDrive\Desktop\beat-ai\model\genre_classifier.pkl")
label_map = joblib.load(r"c:\Users\Abdullah Baran\OneDrive\Desktop\beat-ai\model\label_map.pkl")
scaler = joblib.load(r"c:\Users\Abdullah Baran\OneDrive\Desktop\beat-ai\model\scaler.pkl")

reverse_map = {v: k for k, v in label_map.items()}

def predict(audio_path):
    features = extract_features(audio_path)
    features = scaler.transform([features])
    prediction = model.predict(features)[0]
    return reverse_map[prediction]

app = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload your music"),
    outputs=gr.Label(label="Predicted Genre"),
    title="🎵 beat-ai",
    description="Upload a music file and I'll predict the genre!"
)

app.launch()