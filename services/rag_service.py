import json
import re

from services import chat_service
from services.professional_kb import get_professional_answer
from utils.logger import logger


def _fallback_analysis():
    return {
        "emotional_state_explanation": "Your report suggests moderate emotional strain, with signs that daily stress may be affecting your emotional balance.",
        "stress_indicators": "Important signals include emotional fatigue patterns, stress reactivity, and possible mood imbalance.",
        "recommendations": [
            "Practice daily grounding exercises such as deep breathing or short mindfulness sessions.",
            "Maintain a consistent sleep routine and include light physical activity most days.",
            "Reach out to trusted people and seek professional mental health support if distress persists.",
        ],
    }


def _extract_json_block(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _parse_non_json_response(text):
    cleaned = text.strip()
    if not cleaned:
        return _fallback_analysis()

    emotional_state = ""
    stress_indicators = ""
    recommendations = []

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    current = None
    for line in lines:
        lower = line.lower()
        if "report summary" in lower:
            current = "emotional"
            continue
        if "emotional insights" in lower:
            current = "stress"
            continue
        if "emotional_state_explanation" in lower or "emotional state" in lower:
            current = "emotional"
            continue
        if "stress_indicators" in lower or "stress indicators" in lower:
            current = "stress"
            continue
        if "recommendations" in lower:
            current = "recs"
            continue

        if current == "emotional":
            emotional_state = f"{emotional_state} {line}".strip()
        elif current == "stress":
            stress_indicators = f"{stress_indicators} {line}".strip()
        elif current == "recs":
            recommendation_line = re.sub(r"^[-*\d.\s]+", "", line).strip()
            if recommendation_line:
                recommendations.append(recommendation_line)

    if not emotional_state:
        emotional_state = "The report indicates notable emotional signals that may reflect ongoing stress or emotional burden."
    if not stress_indicators:
        stress_indicators = "Observed signals may include stress reactivity, mood fluctuation, or emotional fatigue."
    if not recommendations:
        recommendations = _fallback_analysis()["recommendations"]

    return {
        "emotional_state_explanation": emotional_state,
        "stress_indicators": stress_indicators,
        "recommendations": recommendations[:5],
    }


def _normalize_result(result):
    if not isinstance(result, dict):
        return _fallback_analysis()

    emotional_state = str(
        result.get("emotional_state_explanation", "")
        or result.get("report_summary", "")
    ).strip()
    stress_indicators = str(
        result.get("stress_indicators", "")
        or result.get("emotional_insights", "")
    ).strip()
    recommendations = result.get("recommendations", [])

    if not isinstance(recommendations, list):
        recommendations = [str(recommendations)] if recommendations else []

    recommendations = [str(item).strip() for item in recommendations if str(item).strip()]

    if not emotional_state:
        emotional_state = _fallback_analysis()["emotional_state_explanation"]
    if not stress_indicators:
        stress_indicators = _fallback_analysis()["stress_indicators"]
    if not recommendations:
        recommendations = _fallback_analysis()["recommendations"]

    return {
        "emotional_state_explanation": emotional_state,
        "stress_indicators": stress_indicators,
        "recommendations": recommendations[:5],
    }


def analyze_report_with_rag(report_text):
    """Analyze generated assessment report with existing RAG context (without storing report in vector DB)."""
    if not report_text or not str(report_text).strip():
        return _fallback_analysis()

    text = str(report_text).strip()

    try:
        if not chat_service.RAG_ENABLED:
            kb_answer = get_professional_answer(f"Emotional guidance based on report: {text[:500]}")
            if kb_answer:
                return {
                    "emotional_state_explanation": "The available report signals indicate emotional stress that should be addressed with supportive daily routines and healthy coping habits.",
                    "stress_indicators": "Possible indicators include emotional fatigue, mood variability, and stress load in current functioning.",
                    "recommendations": [
                        "Use grounding and breathing practices for 10 minutes daily.",
                        "Keep a regular sleep, nutrition, and movement schedule.",
                        "Speak with a counselor or trusted support person if symptoms continue.",
                    ],
                }
            return _fallback_analysis()

        internal_docs = chat_service._retrieve_internal_docs(text)
        external_docs = chat_service._retrieve_external_docs(
            "mental wellness emotional stress coping strategies supportive guidance"
        )
        retrieved_documents = chat_service._build_hybrid_context(internal_docs, external_docs)

        if not retrieved_documents:
            return _fallback_analysis()

        prompt = f"""Context from knowledge base:
{retrieved_documents}

User Emotional Assessment Report:
{text}

Task:
Analyze the emotional indicators present in the report using the mental health knowledge provided in the context.

Important rules:
- Do not provide medical diagnosis.
- Provide supportive, general emotional guidance only.
- Keep tone calm, supportive, and non-judgmental.

Return valid JSON only in this structure:
{{
    "report_summary": "simple explanation of emotional condition",
    "emotional_insights": "important emotional signals and what they may indicate",
    "recommendations": [
        "recommendation 1",
        "recommendation 2",
        "recommendation 3",
        "recommendation 4"
    ]
}}
"""

        tokenizer = getattr(chat_service, "tokenizer", None)
        llm_model = getattr(chat_service, "model", None)
        torch_module = getattr(chat_service, "torch", None)
        if tokenizer is None or llm_model is None or torch_module is None:
            return _fallback_analysis()

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        with torch_module.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=260,
                min_new_tokens=100,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.1,
                do_sample=False,
            )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"RAG report analysis output chars={len(response_text)}")

        json_result = _extract_json_block(response_text)
        if json_result:
            return _normalize_result(json_result)

        return _normalize_result(_parse_non_json_response(response_text))

    except Exception as e:
        logger.error(f"RAG report analysis failed: {str(e)}")
        return _fallback_analysis()
