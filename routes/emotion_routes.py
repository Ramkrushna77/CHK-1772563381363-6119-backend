from fastapi import APIRouter, UploadFile, File
from services.emotion_service import analyze_emotion

router = APIRouter()


async def _detect_emotion(file: UploadFile = File(...)):
    image_bytes = await file.read()

    result = analyze_emotion(image_bytes)

    return {
        "emotion": result.get("emotion"),
        "confidence": result.get("confidence")
    }


@router.post("")
async def detect_emotion_root(file: UploadFile = File(...)):
    return await _detect_emotion(file)


@router.post("/analyze")
async def detect_emotion(file: UploadFile = File(...)):
    return await _detect_emotion(file)