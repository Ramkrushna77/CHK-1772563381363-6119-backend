from fastapi import APIRouter
from pydantic import BaseModel
from services.chat_service import process_chat

router = APIRouter()


class ChatRequest(BaseModel):
    query: str


def _chat_response(request: ChatRequest):
    response = process_chat(request.query)

    return {
        "answer": response["answer"],
        "sentiment": response["sentiment"]
    }


@router.post("")
def chat_with_bot_root(request: ChatRequest):
    return _chat_response(request)


@router.post("/ask")
def chat_with_bot(request: ChatRequest):
    return _chat_response(request)