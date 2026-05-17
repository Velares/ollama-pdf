import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

PDF_DIR = "pdfs"
DB_PATH = "E:/AI/Ollama-PDF/db"

print("🔹 Starting ingestion...")

all_docs = []

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

# Handle empty folder
if not all_docs:
    print("⚠️ No PDFs found in folder. Exiting.")
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
# STEP 3: CREATE EMBEDDINGS
# ==============================
print("🔹 Creating embeddings (fast mode)...")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Ensure DB folder exists
os.makedirs(DB_PATH, exist_ok=True)

# ==============================
# STEP 4: STORE IN DATABASE
# ==============================
db = Chroma.from_documents(
    split_docs,
    embeddings,
    persist_directory=DB_PATH
)

db.persist()

print(f"✅ DONE! Database saved at: {DB_PATH}")