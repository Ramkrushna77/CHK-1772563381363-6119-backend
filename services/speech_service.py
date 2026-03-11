import os
import tempfile
from models.speech_emotion.speech_model import predict_speech_emotion


def analyze_speech(audio_bytes):
    """
    Analyze speech emotion from audio bytes.
    
    Args:
        audio_bytes: Raw audio bytes from uploaded file
        
    Returns:
        dict: Contains 'emotion' and 'confidence' keys or 'error' key if processing fails
    """
    
    if not audio_bytes:
        return {"error": "No audio data provided"}
    
    # Create a temporary file to store audio
    temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
    
    try:
        # Write audio bytes to temp file
        os.write(temp_fd, audio_bytes)
        os.close(temp_fd)
        
        # Predict emotion
        result = predict_speech_emotion(temp_path)
        
        return result
        
    except Exception as e:
        return {"error": f"Failed to process audio: {str(e)}"}
        
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass