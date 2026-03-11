from services.recommendation_engine import generate_recommendations
from services.rag_service import analyze_report_with_rag
import json


def generate_report(face_result, speech_result, sentiment_result):

    face_emotion = face_result.get("emotion", "unknown")
    speech_emotion = speech_result.get("emotion", "unknown")
    sentiment = sentiment_result.get("sentiment", "NEUTRAL")

    recommendations = generate_recommendations(
        face_emotion,
        speech_emotion,
        sentiment
    )

    report = {
        "face_emotion": face_emotion,
        "speech_emotion": speech_emotion,
        "chat_sentiment": sentiment,
        "recommendations": recommendations
    }

    report_text = json.dumps(report, ensure_ascii=True)
    rag_analysis = analyze_report_with_rag(report_text)

    return {
        "report": report,
        "rag_analysis": rag_analysis
    }