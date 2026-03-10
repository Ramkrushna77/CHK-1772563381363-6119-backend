def generate_recommendations(face_emotion, speech_emotion, sentiment):

    recommendations = []

    negative_emotions = ["sad", "angry", "fear", "disgust"]

    if face_emotion in negative_emotions:
        recommendations.append("Try relaxation exercises or meditation.")

    if speech_emotion in negative_emotions:
        recommendations.append("Consider speaking with a trusted friend or counselor.")

    if sentiment == "NEGATIVE":
        recommendations.append("Practice journaling or mindfulness to manage stress.")

    if not recommendations:
        recommendations.append("Keep maintaining your positive mental well-being!")

    return recommendations