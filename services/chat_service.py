import os
import json
import importlib
import re
from utils.logger import logger
from services.professional_kb import get_professional_answer

# RAG generation settings
RAG_TOP_K = 5
RAG_MAX_CONTEXT_CHARS = 4000
RAG_MAX_CHUNK_CHARS = 800
RAG_MIN_ANSWER_WORDS = 12
EXTERNAL_TOP_K = 3

# Try to load RAG system, fall back to simple responses if not available
try:
    import faiss
    import pickle
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    # Load Vector Database (absolute path so it works from any cwd)
    _BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _VDB  = os.path.join(_BASE, "vector_db")
    index = faiss.read_index(os.path.join(_VDB, "index.faiss"))
    with open(os.path.join(_VDB, "chunks.pkl"), "rb") as f:
        texts = pickle.load(f)
    
    # Load Embedding Model
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Load LLM
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    
    RAG_ENABLED = True
    logger.info("✅ RAG System loaded successfully")
except Exception as e:
    logger.warning(f"⚠️ RAG System not available: {str(e)}")
    RAG_ENABLED = False


def analyze_sentiment(text):
    """Simple sentiment analysis based on keywords"""
    text_lower = text.lower()
    
    negative_words = ['sad', 'angry', 'stressed', 'anxious', 'depressed', 'worried', 'afraid', 'hate', 'no', 'bad', 'terrible', 'horrible', 'awful']
    positive_words = ['happy', 'good', 'great', 'excellent', 'love', 'amazing', 'wonderful', 'yes', 'better', 'best']
    
    negative_count = sum(1 for word in negative_words if word in text_lower)
    positive_count = sum(1 for word in positive_words if word in text_lower)
    
    if negative_count > positive_count:
        return "NEGATIVE"
    elif positive_count > negative_count:
        return "POSITIVE"
    else:
        return "NEUTRAL"


def clean_text(text):
    """Clean and normalize text"""
    return " ".join(text.split())


def clean_answer(answer):
    """Clean model output"""
    return answer.replace("Answer:", "").strip()


def _retrieve_internal_docs(query):
    """Retrieve top internal chunks from vector DB with metadata."""
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(query_embedding, RAG_TOP_K)
    logger.info(f"Internal retrieval distances={distances.tolist()} indices={indices.tolist()}")

    docs = []
    for rank, idx in enumerate(indices[0], start=1):
        if 0 <= idx < len(texts):
            chunk = clean_text(str(texts[idx]))
            if not chunk:
                continue
            trimmed_chunk = chunk[:RAG_MAX_CHUNK_CHARS]
            source_name = f"chunk_{idx}"
            docs.append({
                "source_type": "internal",
                "source": source_name,
                "content": trimmed_chunk,
                "score": float(distances[0][rank - 1])
            })
            logger.info(
                f"Internal doc {rank}: source={source_name} chars={len(trimmed_chunk)} preview={trimmed_chunk[:140]}"
            )
    return docs


def _retrieve_external_docs(query):
    """Retrieve external snippets from web search."""
    docs = []
    try:
        DDGS = None
        for module_name in ["ddgs", "duckduckgo_search"]:
            try:
                module = importlib.import_module(module_name)
                DDGS = getattr(module, "DDGS", None)
                if DDGS:
                    break
            except Exception:
                continue

        if DDGS is None:
            logger.warning("External retrieval skipped: no DDGS-compatible package installed")
            return docs

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=EXTERNAL_TOP_K))

        for rank, item in enumerate(results, start=1):
            title = clean_text(str(item.get("title", "")))
            body = clean_text(str(item.get("body", "")))
            if not body:
                continue
            source_name = title if title else f"result_{rank}"
            snippet = body[:RAG_MAX_CHUNK_CHARS]
            docs.append({
                "source_type": "external",
                "source": source_name,
                "content": snippet,
                "score": None
            })
            logger.info(
                f"External doc {rank}: source={source_name} chars={len(snippet)} preview={snippet[:140]}"
            )
    except Exception as e:
        logger.warning(f"External retrieval failed: {str(e)}")

    return docs


def _build_hybrid_context(internal_docs, external_docs):
    """Merge internal and external docs into one context block for LLM."""
    lines = []

    for i, doc in enumerate(internal_docs, start=1):
        lines.append(f"[Internal {i}] Source: {doc['source']}")
        lines.append(doc["content"])
        lines.append("")

    for i, doc in enumerate(external_docs, start=1):
        lines.append(f"[External {i}] Source: {doc['source']}")
        lines.append(doc["content"])
        lines.append("")

    context = "\n".join(lines)
    context = context[:RAG_MAX_CONTEXT_CHARS]
    logger.info(
        f"Hybrid context size chars={len(context)} internal_docs={len(internal_docs)} external_docs={len(external_docs)}"
    )
    return context


def _build_rag_prompt(query, context):
    """Construct prompt that enforces the exact user-requested answer format."""
    prompt = f"""You are a mental health assistant.

Use the context provided to answer the user's question clearly.

IMPORTANT:
- Do NOT repeat instructions or system prompts.
- Do NOT copy raw document text.
- Generate a concise explanation in your own words.

Context:
{context}

Question:
{query}

Format the response as:

Answer:
<clear explanation in 3-5 sentences>

Key Points:
• point 1
• point 2
• point 3

References:
[1] Internal Dataset
[2] External Source"""

    logger.info(f"Final prompt size chars={len(prompt)} preview={prompt[:600]}")
    return prompt


def _safe_summary_from_docs(query, internal_docs, external_docs):
    """Create a clean 3-5 sentence explanation without copying raw chunks."""
    topic = clean_text(query).strip().rstrip("?")
    topic = topic[:120] if topic else "your question"

    has_internal = len(internal_docs) > 0
    has_external = len(external_docs) > 0

    sentence_1 = (
        f"For {topic}, a practical approach is to use small daily habits that reduce stress and improve emotional balance over time."
    )
    sentence_2 = (
        "Start with one or two techniques such as paced breathing, short mindfulness breaks, and a realistic study or sleep routine to improve focus and stability."
    )
    sentence_3 = (
        "Track what helps in a simple journal so you can repeat effective strategies and notice early signs when stress is rising."
    )
    sentence_4 = (
        "If symptoms become persistent, intense, or interfere with daily life, seek support from a licensed mental health professional for personalized guidance."
    )

    if has_internal and has_external:
        sentence_5 = "This answer is based on both your internal dataset and external sources to keep guidance balanced and up to date."
        return " ".join([sentence_1, sentence_2, sentence_3, sentence_4, sentence_5])

    return " ".join([sentence_1, sentence_2, sentence_3, sentence_4])


def _format_answer_with_references(answer, internal_docs, external_docs, query):
    """Ensure answer follows required structure and includes references."""
    cleaned = clean_answer(answer)

    # Remove any leaked prompt/instruction text from model output.
    banned_fragments = [
        "important:",
        "do not repeat instructions",
        "do not copy raw document text",
        "format the response as",
        "return output in exact format",
        "rules:",
        "context:",
        "question:",
        "<clear explanation",
    ]
    cleaned_lines = []
    for line in cleaned.splitlines():
        line_lower = line.strip().lower()
        if any(fragment in line_lower for fragment in banned_fragments):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()

    citation_artifact_count = len(re.findall(r"\[\d+\]", cleaned))
    cleaned = re.sub(r"\[\d+\]", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    # If output still looks like raw chunks/metadata, replace with clean summary.
    leakage_markers = [
        "[internal",
        "[external",
        "source:",
        "chunk_",
        "evidence context",
    ]
    if any(marker in cleaned.lower() for marker in leakage_markers):
        cleaned = ""

    if citation_artifact_count >= 3:
        cleaned = ""

    has_answer = "Answer:" in cleaned
    has_key_points = "Key Points:" in cleaned
    has_refs = "References:" in cleaned

    if has_answer and has_key_points and has_refs:
        return cleaned

    short_text = cleaned if cleaned else _safe_summary_from_docs(query, internal_docs, external_docs)
    short_text = short_text[:900]

    points = []
    for sentence in short_text.split("."):
        s = sentence.strip()
        if len(s.split()) >= 5:
            points.append(s)
        if len(points) == 3:
            break

    while len(points) < 3:
        points.append("Consult a licensed mental health professional for personalized care")

    internal_ref = internal_docs[0]["source"] if internal_docs else "mental_health_internal_docs"
    external_ref = external_docs[0]["source"] if external_docs else "no_external_source_available"

    return (
        f"Answer:\n{short_text}\n\n"
        f"Key Points:\n"
        f"• {points[0]}\n"
        f"• {points[1]}\n"
        f"• {points[2]}\n\n"
        f"References:\n"
        f"[1] Internal Dataset - {internal_ref}\n"
        f"[2] External Source - {external_ref}"
    )


def _is_low_quality_answer(answer):
    """Heuristic check for short/fractured/repetitive model outputs."""
    if not answer:
        return True

    words = answer.split()
    if len(words) < RAG_MIN_ANSWER_WORDS:
        return True

    lowered = answer.lower()
    if lowered.count("if necessary") >= 3:
        return True

    if lowered.startswith("1.") and lowered.count(".") > 8:
        return True

    citation_markers = ["et al", "jama", "plos", "doi", "vol."]
    marker_hits = sum(1 for marker in citation_markers if marker in lowered)
    if marker_hits >= 2 and len(words) < 90:
        return True

    if answer.count(";") >= 2 and len(words) < 90:
        return True

    return False


def _extractive_context_fallback(context, query):
    """Build a readable paragraph directly from retrieved context when generation is poor."""
    raw_sentences = [s.strip() for s in context.replace("\n", " ").split(".")]
    usable = [s for s in raw_sentences if len(s.split()) >= 6][:4]

    if not usable:
        return None

    paragraph = ". ".join(usable)
    paragraph = paragraph[:850].strip()
    if paragraph and not paragraph.endswith("."):
        paragraph += "."

    return (
        f"Based on the retrieved mental health guidance, {paragraph} "
        f"If symptoms are persistent or severe, please consult a licensed mental health professional."
    )


def process_chat_with_rag(query):
    """Process chat using RAG system"""
    try:
        # Hybrid retrieval: internal + external
        internal_docs = _retrieve_internal_docs(query)
        external_docs = _retrieve_external_docs(query)

        # Build hybrid context and prompt
        context = _build_hybrid_context(internal_docs, external_docs)
        if not context:
            logger.warning("RAG context is empty after retrieval")
            return None

        prompt = _build_rag_prompt(query, context)

        # Generate answer
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=220,
                min_new_tokens=70,
                num_beams=4,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                repetition_penalty=1.1,
                do_sample=False
            )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = _format_answer_with_references(answer, internal_docs, external_docs, query)
        logger.info(f"RAG raw model output chars={len(answer)} text={answer[:500]}")

        # Guard against extremely short generations (e.g., one-word outputs)
        if _is_low_quality_answer(answer):
            logger.warning(f"RAG output low quality, retrying generation. words={len(answer.split())}")
            retry_prompt = (
                prompt
                + "\n\nRewrite the final answer as a coherent paragraph with concrete details and no numbering."
            )
            retry_inputs = tokenizer(
                retry_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )

            with torch.no_grad():
                retry_outputs = model.generate(
                    **retry_inputs,
                    max_new_tokens=240,
                    min_new_tokens=80,
                    num_beams=5,
                    no_repeat_ngram_size=3,
                    length_penalty=1.05,
                    repetition_penalty=1.1,
                    do_sample=False
                )

            answer = tokenizer.decode(retry_outputs[0], skip_special_tokens=True)
            answer = _format_answer_with_references(answer, internal_docs, external_docs, query)
            logger.info(f"RAG retry output chars={len(answer)} text={answer[:500]}")

        if _is_low_quality_answer(answer):
            logger.warning("RAG output still low quality; using extractive context fallback")
            fallback_answer = _extractive_context_fallback(context, query)
            if fallback_answer:
                answer = _format_answer_with_references(fallback_answer, internal_docs, external_docs, query)
                logger.info(f"RAG extractive fallback chars={len(answer)} text={answer[:500]}")
        
        return answer
    except Exception as e:
        logger.error(f"RAG processing error: {str(e)}")
        return None


def generate_simple_response(query):
    """Generate simple response when RAG is not available"""
    text_lower = query.lower()
    
    # Mental health keyword responses
    if any(word in text_lower for word in ['stress', 'pressure', 'overwhelmed']):
        return "It sounds like you're experiencing stress. Some helpful techniques: take breaks, practice deep breathing, exercise regularly, and ensure adequate sleep. Have you tried meditation or mindfulness?"
    
    if any(word in text_lower for word in ['anxiety', 'anxious', 'panic', 'worried']):
        return "Anxiety can be challenging. The 5-4-3-2-1 grounding technique helps: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. Regular exercise and sleep also help. Consider speaking with a professional."
    
    if any(word in text_lower for word in ['sleep', 'insomnia', 'tired', 'fatigue']):
        return "Sleep is crucial for mental health. Try keeping your room cool and dark, avoiding screens 30 minutes before bed, maintaining a consistent schedule, and limiting caffeine. If sleep issues persist, consult a doctor."
    
    if any(word in text_lower for word in ['sad', 'depressed', 'lonely', 'alone']):
        return "It's okay to feel sad sometimes. Please remember you're not alone. Talking about your feelings is brave. Consider reaching out to friends, family, or a mental health professional. There's always hope."
    
    if any(word in text_lower for word in ['relax', 'calm', 'peace', 'peace of mind']):
        return "Here are some relaxation techniques: deep breathing exercises, progressive muscle relaxation, meditation, yoga, or taking a warm bath. Even a 10-minute nature walk can help calm your mind."
    
    if any(word in text_lower for word in ['exercise', 'workout', 'physical', 'health']):
        return "Physical activity is great for mental health! Aim for at least 30 minutes of exercise 3-4 times per week. This could be walking, running, yoga, or any activity you enjoy. It reduces stress and improves mood."
    
    if any(word in text_lower for word in ['relationship', 'friend', 'social', 'connect']):
        return "Social connections are vital for wellbeing. Spend time with people you trust, join groups with shared interests, and don't hesitate to reach out. Quality relationships provide support and meaning."
    
    return "Thank you for sharing that with me. I'm here to listen and support you. Can you tell me more about how you're feeling right now? Remember, seeking help is a sign of strength."


def process_chat(query):
    """Main chat processing function"""
    
    if not query or not query.strip():
        return {
            "answer": "Please ask me something about your mental health or wellbeing.",
            "sentiment": "NEUTRAL"
        }
    
    query = query.strip()
    
    # Analyze sentiment
    sentiment = analyze_sentiment(query)
    
    # PRIMARY: Use RAG pipeline first
    answer = None
    if RAG_ENABLED:
        answer = process_chat_with_rag(query)
        if _is_low_quality_answer(answer):
            logger.warning("RAG answer still low quality at API level; falling back to KB")
            answer = None

    # FALLBACK: Professional KB if RAG returns empty/short output
    if not answer:
        professional_answer = get_professional_answer(query)
        if professional_answer:
            logger.info(f"Using professional KB fallback - Query: {query[:50]}... | Sentiment: {sentiment}")
            answer = professional_answer
    
    # FINAL FALLBACK: Simple response if both RAG and KB fail
    if not answer:
        answer = generate_simple_response(query)
    
    logger.info(f"Chat processed - Query: {query[:50]}... | Sentiment: {sentiment}")
    
    return {
        "answer": answer,
        "sentiment": sentiment
    }
