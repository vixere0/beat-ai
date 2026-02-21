import librosa
import numpy as np

def extract_features(audio_path):
    y, sr = librosa.load(audio_path)

   
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.squeeze(tempo))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

 
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    return np.concatenate([
        mfcc_mean, mfcc_std, chroma_mean,
        [spec_centroid, zcr, float(tempo)]
    ])
    
