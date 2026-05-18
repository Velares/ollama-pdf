import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

PDF_DIR = "pdfs"
PROCESSED_DIR = "processed"
DB_PATH = "E:/AI/Ollama-PDF/db"

print("🔹 Incremental ingestion starting...")

# Load embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load existing DB (DO NOT recreate)
db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

if not files:
    print("⚠️ No new PDFs found.")
    exit()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=40
)

for file in files:
    file_path = os.path.join(PDF_DIR, file)

    print(f"📄 Processing: {file}")

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # ✅ ADD METADATA HERE
    for doc in docs:
        doc.metadata["source"] = file
        doc.metadata["page"] = doc.metadata.get("page", "unknown")

    split_docs = text_splitter.split_documents(docs)

    print(f"✅ {len(split_docs)} chunks created")

    # ✅ ADD TO EXISTING DB
    db.add_documents(split_docs)

    # move file after processing
    dst = os.path.join(PROCESSED_DIR, file)
    shutil.move(file_path, dst)

    print(f"📦 Moved: {file} → processed/")

db.persist()

print("✅ Incremental ingestion complete!")