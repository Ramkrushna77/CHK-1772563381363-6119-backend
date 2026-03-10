import librosa
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "speech_emotion_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

model = None
scaler = None
label_encoder = None
SPEECH_MODEL_READY = False

try:
    model = load_model(MODEL_PATH)
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    label_encoder = pickle.load(open(ENCODER_PATH, "rb"))
    SPEECH_MODEL_READY = True
except Exception:
    SPEECH_MODEL_READY = False


def extract_features(file_path):

    audio, sample_rate = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc


def predict_speech_emotion(file_path):

    if not SPEECH_MODEL_READY:
        return {
            "emotion": "neutral",
            "confidence": 0.0
        }

    features = extract_features(file_path)

    features = scaler.transform([features])

    prediction = model.predict(features)

    emotion_index = np.argmax(prediction)

    emotion = label_encoder.inverse_transform([emotion_index])[0]

    confidence = float(np.max(prediction))

    return {
        "emotion": emotion,
        "confidence": confidence
    }