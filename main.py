from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routes
from routes.chat_routes import router as chat_router
from routes.emotion_routes import router as emotion_router
from routes.speech_routes import router as speech_router
from routes.report_routes import router as report_router

from utils.logger import logger


# --------------------------------------------------
# Create FastAPI App
# --------------------------------------------------

app = FastAPI(
    title="Mental Health AI Platform",
    description="Backend API for Emotion Detection, Speech Analysis, and RAG Chatbot",
    version="1.0.0"
)


# --------------------------------------------------
# CORS Middleware (Required for React Frontend)
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Health Check Endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    logger.info("Health check called")
    return {"status": "Backend running"}


# --------------------------------------------------
# Include API Routes
# --------------------------------------------------

app.include_router(
    chat_router,
    prefix="/chat",
    tags=["Chatbot"]
)

app.include_router(
    emotion_router,
    prefix="/emotion",
    tags=["Facial Emotion"]
)

app.include_router(
    speech_router,
    prefix="/speech",
    tags=["Speech Emotion"]
)

app.include_router(
    report_router,
    prefix="/report",
    tags=["Report Generator"]
)


# --------------------------------------------------
# Root Endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Mental Health AI Backend Running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)