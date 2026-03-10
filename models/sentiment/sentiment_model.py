from transformers import pipeline

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


def analyze_sentiment(text):

    result = sentiment_pipeline(text)[0]

    label = result["label"]
    score = float(result["score"])

    return {
        "sentiment": label,
        "confidence": score
    }