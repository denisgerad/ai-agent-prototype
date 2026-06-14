#PDF + Tools + Agent + Debug Agent
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.tools import BaseTool, tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
import requests
import sys
import os

# Add agents directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))
from debug_agent import DebugAgent

# ---------- CONFIG ----------
PDF_PATH = "sample.pdf"
OLLAMA_MODEL = "mistral"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"
# ---------------------------





def main():
    print("[*] Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print("[*] Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    print("[*] Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("[*] Creating vector store...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print("[*] Initializing LLM...")
    llm = ChatOllama(model=OLLAMA_MODEL)

    # ---------- PDF QA ----------
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    
    @tool
    def pdf_search(question: str) -> str:
        """Use this to answer questions about the uploaded PDF document. Input should be a clear question about the PDF content."""
        docs = retriever.invoke(question)
        context = "\n".join([d.page_content for d in docs])
        prompt = f"Based on this context:\n{context}\n\nAnswer this question: {question}"
        response = llm.invoke(prompt)
        return response

    @tool
    def weather(city: str) -> str:
        """Get current weather for a city. Input should be a city name."""
        resp = requests.get(f"https://wttr.in/{city}?format=3")
        return resp.text

    @tool
    def web_scraper(url: str) -> str:
        """Scrape text content from a webpage. Input should be a valid URL starting with http:// or https://"""
        loader = WebBaseLoader(url)
        docs = loader.load()
        return "\n".join([d.page_content[:1000] for d in docs])

    # ---------- TOOLS ----------
    tools = [pdf_search, weather, web_scraper]

    # Create agent using LangGraph
    from langchain_core.messages import HumanMessage
    
    agent_executor = create_react_agent(llm, tools)

    print("\n[+] Agent ready! Ask anything (type 'exit' to quit)\n")
    
    while True:
        query = input("[?] Question: ")
        
        if query.lower() == "exit":
            break
        
        try:
            # Invoke agent with the query
            response = agent_executor.invoke({"messages": [HumanMessage(content=query)]})
            print("\n[*] Answer:\n", response.get("messages", [])[-1].content if response.get("messages") else response, "\n")
        except Exception as e:
            print(f"\n[!] Error: {str(e)}\n")


if __name__ == "__main__":
    main()
