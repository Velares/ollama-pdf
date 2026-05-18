from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM

DB_PATH = "E:/AI/Ollama-PDF/db"

print("🔹 Loading vector database...")

# Load embedding model (MUST match ingest)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load vector database
db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

# Load language model for answering questions
llm = OllamaLLM(model="llama3")

print("✅ Ready! Ask questions about your PDFs (type 'exit' to quit)")

while True:
    query = input("\n❓ Question: ")

    if query.lower() == "exit":
        print("👋 Exiting...")
        break

    # Retrieve relevant documents
    docs = db.similarity_search(query, k=4)

    if not docs:
        print("⚠️ No relevant context found.")
        continue

    # Build context from retrieved chunks
    context = "\n".join([doc.page_content for doc in docs])

    # Create prompt
    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""

    # Get response
    response = llm.invoke(prompt)

    print("\n✅ Answer:\n")
    print(response)

    # ==============================
    # SHOW SOURCES (METADATA)
    # ==============================
    print("\n📚 Sources:\n")

    seen = set()

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown file")
        page = doc.metadata.get("page", "Unknown page")

        key = f"{source}-{page}"

        # Avoid duplicate listings
        if key in seen:
            continue
        seen.add(key)

        print(f"[{i}] {source} (Page {page})")