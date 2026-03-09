import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load vector database
index = faiss.read_index("../vector_db/index.faiss")

with open("../vector_db/chunks.pkl", "rb") as f:
    texts = pickle.load(f)

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load LLM model
generator = pipeline("text-generation", model="google/flan-t5-base")


while True:

    query = input("\nAsk a question: ")

    # Convert query into embedding
    query_embedding = embed_model.encode([query])

    # Retrieve top 5 relevant chunks
    distances, indices = index.search(query_embedding, 5)

    # Combine retrieved chunks into context
    context = ""

    for i in indices[0]:
        context += texts[i] + "\n"

    # Better prompt for summarization
    prompt = f"""
You are a helpful mental health assistant.

Using the context below, answer the user's question clearly and briefly.

Context:
{context}

Question:
{query}

Provide a short summarized answer in bullet points.
"""

    # Generate answer
    result = generator(
        prompt,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.3
    )

    answer = result[0]["generated_text"]

    print("\nAnswer:\n")
    print(answer)