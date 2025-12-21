
#working code for RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA

# -------- CONFIG --------
PDF_PATH = "sample.pdf"
OLLAMA_MODEL = "mistral"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"
# ------------------------

def main():
    print("📄 Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print("✂️ Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    print("🔢 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("📦 Creating vector store...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print("🤖 Initializing LLM...")
    llm = OllamaLLM(model=OLLAMA_MODEL)

    print("🔍 Creating RetrievalQA chain...")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )

    print("\n✅ Ready! Ask questions (type 'exit' to quit)\n")

    while True:
        query = input("❓ Question: ")
        if query.lower() == "exit":
            break

        result = qa(query)
        print("\n💡 Answer:\n", result["result"], "\n")

if __name__ == "__main__":
    main()
