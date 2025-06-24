# 🎙️ Speech Emotion Recognition using Deep Learning

This project is an end-to-end system that classifies human emotions from speech using a trained deep learning model. It leverages audio feature extraction techniques (MFCC, Chroma, Mel Spectrogram, etc.) and a neural network to predict emotions from audio input.

## 📁 Project Structure

```
Emotion-Classification/
├── model/                     # Trained model
│   └── emotion_model.h5
├── notebook/
│   └── Emotion_Classification.ipynb
├── scripts/                   # CLI-based testing
│   └── test_model.py
├── streamlit_app/            # Streamlit web application
│   └── app.py
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── demo.mp4                  # Optional demo video (to be added)
```

## 🧠 Model Architecture

- Input: Extracted features (MFCC, Chroma, Mel, Contrast, Tonnetz)
- Architecture: Dense Neural Network
  - Dense(512) → ReLU → Dropout(0.3)
  - Dense(256) → ReLU → Dropout(0.3)
  - Dense(128) → ReLU → Dropout(0.3)
  - Output: Softmax over 8 emotion classes
- Trained with: Adam optimizer, early stopping on validation accuracy

### 🎯 Target Emotions
- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

---

## 🧪 Evaluation Metrics

| Metric               | Value (Target) |
|----------------------|----------------|
| Overall Accuracy     | > 80% ✅         |
| F1 Score (Macro Avg) | > 80% ✅         |
| Per-Class Accuracy   | > 75% ✅         |
| Confusion Matrix     | ✅ Included     |

All metrics are calculated and visualized in the notebook.

---

## 🚀 How to Run

### 1. 🔧 Setup Environment

Install dependencies using:

```bash
pip install -r requirements.txt
```

### 2. 🧪 Run Command-Line Prediction

Test the model on any `.wav` file using:

```bash
python scripts/test_model.py path/to/audio.wav
```

### 3. 🌐 Run the Streamlit Web App

```bash
cd streamlit_app
streamlit run app.py
```

Then open: [http://localhost:8501](http://localhost:8501)

### 4. 📥 Upload Audio File

Upload a short `.wav` file. The app will return the predicted emotion like:

```
Predicted Emotion: Happy 🎯
```


## 🤝 Credits

- Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- Libraries: TensorFlow, Librosa, Streamlit, Scikit-learn

---
