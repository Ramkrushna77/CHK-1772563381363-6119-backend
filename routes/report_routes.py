from fastapi import APIRouter
from pydantic import BaseModel
from services.report_generator import generate_report

router = APIRouter()


class ReportRequest(BaseModel):
    face_emotion: str
    speech_emotion: str
    sentiment: str


def _generate_user_report(data: ReportRequest):
    report = generate_report(
        {"emotion": data.face_emotion},
        {"emotion": data.speech_emotion},
        {"sentiment": data.sentiment}
    )

    return report


@router.post("")
def generate_user_report_root(data: ReportRequest):
    return _generate_user_report(data)


@router.post("/generate")
def generate_user_report(data: ReportRequest):
    return _generate_user_report(data)