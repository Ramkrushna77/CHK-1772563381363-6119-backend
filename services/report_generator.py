from services.recommendation_engine import generate_recommendations


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

    return report