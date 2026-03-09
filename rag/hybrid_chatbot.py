import faiss
import pickle
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ddgs import DDGS
import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------
# Load Vector Database
# --------------------------------------------------

try:
    index = faiss.read_index("../vector_db/index.faiss")

    with open("../vector_db/chunks.pkl", "rb") as f:
        texts = pickle.load(f)

    print("✅ Vector database loaded")

except Exception as e:
    print("Error loading vector DB:", e)
    exit()


# --------------------------------------------------
# Load Embedding Model
# --------------------------------------------------

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("✅ Embedding model loaded")


# --------------------------------------------------
# Load LLM
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

print("✅ Language model loaded")


# --------------------------------------------------
# Web Search
# --------------------------------------------------

def search_web(query):

    results = []

    try:

        with DDGS() as ddgs:

            for r in ddgs.text(query, max_results=2):

                if "body" in r:
                    results.append(r["body"])

    except:
        pass

    return "\n".join(results)


# --------------------------------------------------
# Clean Context
# --------------------------------------------------

def clean_text(text):

    return " ".join(text.split())


# --------------------------------------------------
# Build Context
# --------------------------------------------------

def build_context(pdf_context, web_context):

    pdf_context = clean_text(pdf_context)
    web_context = clean_text(web_context)

    if web_context:
        combined = pdf_context + "\n\nAdditional web information:\n" + web_context
    else:
        combined = pdf_context

    return combined[:1500]


# --------------------------------------------------
# Clean Model Output
# --------------------------------------------------

def clean_answer(answer):

    return answer.replace("Answer:", "").strip()


# --------------------------------------------------
# Chat Interface
# --------------------------------------------------

print("\n==============================================")
print("🧠 AI Mental Health Assistant (Hybrid RAG)")
print("==============================================")
print("Type 'exit' to stop\n")


while True:

    query = input("🔹 Ask a question: ").strip()

    if query.lower() == "exit":
        break

    if query == "":
        print("Please enter a question.")
        continue


    print("\n⏳ Thinking...\n")


# --------------------------------------------------
# Create Query Embedding
# --------------------------------------------------

    query_embedding = embed_model.encode([query])


# --------------------------------------------------
# Vector Search
# --------------------------------------------------

    distances, indices = index.search(query_embedding, 3)

    pdf_context = ""

    for i in indices[0]:

        if i < len(texts):
            pdf_context += texts[i] + "\n"


# --------------------------------------------------
# Web Search
# --------------------------------------------------

    web_context = search_web(query)


# --------------------------------------------------
# Combine Context
# --------------------------------------------------

    context = build_context(pdf_context, web_context)


# --------------------------------------------------
# Prompt (Improved)
# --------------------------------------------------

    prompt = f"""
You are an expert mental health assistant trained to explain mental health topics clearly.

Use the information below to answer the question.

Context:
{context}

Question:
{query}

Provide a detailed response that includes:

• A short explanation of the topic  
• Key symptoms, causes, or factors if relevant  
• Helpful advice or management strategies  
• When someone should seek professional help  

Write the answer in a clear and structured way.
"""


# --------------------------------------------------
# Tokenize
# --------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )


# --------------------------------------------------
# Generate Answer
# --------------------------------------------------

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=200,

            temperature=0.5,

            top_p=0.9,

            repetition_penalty=1.3,

            do_sample=True
        )


# --------------------------------------------------
# Decode Answer
# --------------------------------------------------

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    answer = clean_answer(answer)


# --------------------------------------------------
# Print Answer
# --------------------------------------------------

    print("\n🤖 AI Assistant:\n")

    print(answer)

    print("\n--------------------------------------------\n")