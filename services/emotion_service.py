import cv2
import numpy as np
from models.facial_emotion.face_model import predict_emotion
from utils.face_detection import detect_face


def analyze_emotion(image_bytes):

    # Convert bytes → numpy image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    # Detect face
    face = detect_face(img)

    if face is None:
        return {"error": "No face detected"}

    # Predict emotion
    result = predict_emotion(face)

    return result