from fastapi import APIRouter, UploadFile, File
from services.speech_service import analyze_speech

router = APIRouter()


async def _detect_speech_emotion(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    result = analyze_speech(audio_bytes)

    return {
        "emotion": result.get("emotion"),
        "confidence": result.get("confidence")
    }


@router.post("")
async def detect_speech_emotion_root(file: UploadFile = File(...)):
    return await _detect_speech_emotion(file)


@router.post("/analyze")
async def detect_speech_emotion(file: UploadFile = File(...)):
    return await _detect_speech_emotion(file)