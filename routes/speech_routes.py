from fastapi import APIRouter, UploadFile, File, HTTPException
from services.speech_service import analyze_speech

router = APIRouter()


@router.post("")
async def detect_speech_emotion_root(file: UploadFile = File(...)):
    """Detect speech emotion from uploaded audio file."""
    try:
        audio_bytes = await file.read()
        
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Audio file is empty")
        
        result = analyze_speech(audio_bytes)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "emotion": result.get("emotion"),
            "confidence": result.get("confidence")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@router.post("/analyze")
async def detect_speech_emotion(file: UploadFile = File(...)):
    """Alternative endpoint for speech emotion detection."""
    try:
        audio_bytes = await file.read()
        
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Audio file is empty")
        
        result = analyze_speech(audio_bytes)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "emotion": result.get("emotion"),
            "confidence": result.get("confidence")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")