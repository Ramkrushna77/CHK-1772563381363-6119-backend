import faiss
import pickle
from sentence_transformers import SentenceTransformer


# Load vector database
index = faiss.read_index("../vector_db/index.faiss")

with open("../vector_db/chunks.pkl", "rb") as f:
    texts = pickle.load(f)


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


while True:

    query = input("\nAsk a question: ")

    query_embedding = model.encode([query])

    distances, indices = index.search(query_embedding, 3)

    print("\nTop relevant information:\n")

    for i in indices[0]:
        print(texts[i])
        print("\n------------------\n")