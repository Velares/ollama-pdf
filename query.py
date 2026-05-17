from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM

DB_PATH = "E:/AI/Ollama-PDF/db"

print("🔹 Loading vector database...")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

llm = OllamaLLM(model="llama3")

print("✅ Ready! Ask questions about your PDFs (type 'exit' to quit)")

while True:
    query = input("\n❓ Question: ")

    if query.lower() == "exit":
        print("👋 Exiting...")
        break

    docs = db.similarity_search(query, k=4)

    if not docs:
        print("⚠️ No relevant context found.")
        continue

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    print("\n✅ Answer:\n")
    print(response)
