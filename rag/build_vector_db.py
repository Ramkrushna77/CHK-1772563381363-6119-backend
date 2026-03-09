import os
import pickle
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


folder_path = "../data/knowledge_base"

documents = []

# -------------------------
# Load PDFs
# -------------------------
for file in os.listdir(folder_path):

    if file.endswith(".pdf"):

        file_path = os.path.join(folder_path, file)

        try:
            reader = PdfReader(file_path)

            for page in reader.pages:

                text = page.extract_text()

                if text:
                    documents.append(text)

        except Exception as e:
            print(f"Skipping {file}: {e}")

print("Total pages loaded:", len(documents))


# -------------------------
# Split text into chunks
# -------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.create_documents(documents)

texts = [doc.page_content for doc in chunks]

print("Total chunks:", len(texts))


# -------------------------
# Create embeddings
# -------------------------
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(texts)

print("Embeddings created:", len(embeddings))


# -------------------------
# Create FAISS index
# -------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

print("Vectors stored in FAISS:", index.ntotal)


# -------------------------
# Save database
# -------------------------
os.makedirs("../vector_db", exist_ok=True)

faiss.write_index(index, "../vector_db/index.faiss")

with open("../vector_db/chunks.pkl", "wb") as f:
    pickle.dump(texts, f)

print("Vector database saved successfully!")