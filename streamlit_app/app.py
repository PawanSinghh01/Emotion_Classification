import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("C:/Users/DELL/Desktop/mars ML/model/emotion_model.h5", compile=False)
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

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

st.title("🎙️ Speech Emotion Recognizer")
st.markdown("Upload an audio file and the app will predict the emotion.")

uploaded_file = st.file_uploader("Choose a .wav audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    try:
        features = extract_features("temp.wav").reshape(1, -1)
        prediction = model.predict(features)
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        st.success(f"Predicted Emotion: **{predicted_emotion.capitalize()}** 🎯")
    except Exception as e:
        st.error(f"Error: {str(e)}")
