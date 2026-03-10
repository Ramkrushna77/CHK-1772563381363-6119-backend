import cv2
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "emotion_model.h5")

# Load model; wrap in try/except so a TF/Keras version mismatch
# does not crash the whole server at startup.
try:
    from tensorflow.keras.models import load_model
    emotion_model = load_model(MODEL_PATH)
except Exception as _e:
    print(f"[face_model] WARNING: could not load emotion model: {_e}")
    emotion_model = None

emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


def preprocess_face(face_img):
    face = cv2.resize(face_img, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face / 255.0
    face = np.reshape(face, (1, 48, 48, 1))
    return face


def predict_emotion(face_img):
    if emotion_model is None:
        return {"emotion": "neutral", "confidence": 0.5}

    processed = preprocess_face(face_img)

    prediction = emotion_model.predict(processed)
    emotion_index = np.argmax(prediction)

    emotion = emotion_labels[emotion_index]
    confidence = float(np.max(prediction))

    return {
        "emotion": emotion,
        "confidence": confidence
    }