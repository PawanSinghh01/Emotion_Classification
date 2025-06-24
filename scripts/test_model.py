# scripts/test_model.py

import librosa
import numpy as np
from tensorflow.keras.models import load_model
import sys

# Emotion classes
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Load the trained model
model = load_model("model/emotion_model.h5")

# Feature extractor
def extract_features(file_path):
    X, sr = librosa.load(file_path, sr=None, duration=3, res_type='kaiser_fast')
    result = []

    mfcc = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40)
    result.extend(np.mean(mfcc.T, axis=0))

    chroma = librosa.feature.chroma_stft(y=X, sr=sr)
    result.extend(np.mean(chroma.T, axis=0))

    mel = librosa.feature.melspectrogram(y=X, sr=sr)
    result.extend(np.mean(mel.T, axis=0))

    contrast = librosa.feature.spectral_contrast(y=X, sr=sr)
    result.extend(np.mean(contrast.T, axis=0))

    y_harm = librosa.effects.harmonic(X)
    tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
    result.extend(np.mean(tonnetz.T, axis=0))

    return np.array(result)

# Entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <path_to_audio.wav>")
        sys.exit(1)

    file_path = sys.argv[1]
    features = extract_features(file_path)
    features = features.reshape(1, -1)

    prediction = model.predict(features)
    predicted_class = emotion_labels[np.argmax(prediction)]

    print(f"Predicted Emotion: {predicted_class}")
