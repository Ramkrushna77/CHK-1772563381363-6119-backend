import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
folder_path = "../data/knowledge_base"

documents = []

# Step 1: Load PDFs manually
for file in os.listdir(folder_path):

    if file.endswith(".pdf"):

        file_path = os.path.join(folder_path, file)

        reader = PdfReader(file_path)

        for page in reader.pages:
            text = page.extract_text()

            if text:
                documents.append(text)


print("Total pages loaded:", len(documents))



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.create_documents(documents)

print("Total chunks created:", len(chunks))