import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

PDF_DIR = "pdfs"
PROCESSED_DIR = "processed"
DB_PATH = "E:/AI/Ollama-PDF/db"

print("🔹 Starting ingestion...")

all_docs = []
processed_files = []

# ==============================
# STEP 1: LOAD ALL PDFs
# ==============================
for file in os.listdir(PDF_DIR):
    if file.lower().endswith(".pdf"):
        file_path = os.path.join(PDF_DIR, file)
        print(f"📄 Loading: {file}")

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        all_docs.extend(docs)
        processed_files.append(file)

# Handle empty folder
if not all_docs:
    print("⚠️ No PDFs found. Nothing to process.")
    exit()

print(f"✅ Loaded {len(all_docs)} total pages")

# ==============================
# STEP 2: SPLIT INTO CHUNKS
# ==============================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=40
)

split_docs = text_splitter.split_documents(all_docs)

print(f"✅ Split into {len(split_docs)} chunks")

# ==============================
# STEP 3: EMBEDDINGS (FAST)
# ==============================
print("🔹 Creating embeddings...")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Ensure DB exists
os.makedirs(DB_PATH, exist_ok=True)

# ==============================
# STEP 4: STORE IN DB
# ==============================
db = Chroma.from_documents(
    split_docs,
    embeddings,
    persist_directory=DB_PATH
)

db.persist()

print("✅ Ingestion complete!")

# ==============================
# STEP 5: MOVE PROCESSED FILES
# ==============================
os.makedirs(PROCESSED_DIR, exist_ok=True)

for file in processed_files:
    src = os.path.join(PDF_DIR, file)
    dst = os.path.join(PROCESSED_DIR, file)

    shutil.move(src, dst)
    print(f"📦 Moved: {file} → processed/")

print("✅ All files moved to processed folder.")