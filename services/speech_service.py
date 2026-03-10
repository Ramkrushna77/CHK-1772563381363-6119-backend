import os
from models.speech_emotion.speech_model import predict_speech_emotion


def analyze_speech(audio_file):

    # Save uploaded file temporarily
    temp_path = "temp_audio.wav"

    with open(temp_path, "wb") as f:
        f.write(audio_file)

    try:
        result = predict_speech_emotion(temp_path)

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return result