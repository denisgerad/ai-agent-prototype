#PDF + Tools + Agent
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_classic.tools import Tool
from langchain_classic.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
import requests

# ---------- CONFIG ----------
PDF_PATH = "sample.pdf"
OLLAMA_MODEL = "mistral"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"
# ---------------------------


# ---------- CUSTOM TOOLS ----------
def get_weather(city: str) -> str:
    """Get current weather for a city"""
    resp = requests.get(f"https://wttr.in/{city}?format=3")
    return resp.text

def scrape_website(url: str) -> str:
    """Scrape text content from a webpage"""
    loader = WebBaseLoader(url)
    docs = loader.load()
    return "\n".join([d.page_content[:1000] for d in docs])
# -------------------------------


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

    # ---------- PDF QA ----------
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=False
    )

    # ---------- TOOLS ----------
    pdf_tool = Tool(
        name="PDF_Search",
        func=qa.invoke,
        description="Use this to answer questions about the uploaded PDF document. Input should be a clear question about the PDF content."
    )

    weather_tool = Tool(
        name="Weather",
        func=get_weather,
        description="Get current weather for a city. Input should be a city name."
    )

    web_scraper_tool = Tool(
        name="Web_Scraper",
        func=scrape_website,
        description="Scrape text content from a webpage. Input should be a valid URL starting with http:// or https://"
    )

    tools = [pdf_tool, weather_tool, web_scraper_tool]

    # ---------- MEMORY ----------
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output"
    )

    # ---------- AGENT ----------
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=5,
        memory=memory,
        agent_kwargs={
            "prefix": """Assistant is a helpful AI that can use tools to answer questions. When the user asks a follow-up question, always consider the previous conversation context to understand what they're referring to. If you need to search for information, include the relevant topic from the conversation history in your search query.

For example:
- If the conversation is about "one tap trading" and the user asks "list popular apps", search for "popular one tap trading apps".
- If discussing a city and asked "what's the temperature", use that city name.

Always provide the most relevant and contextual answer based on the full conversation."""
        }
    )

    print("\n✅ Agent ready with conversation memory! Ask anything (type 'exit' to quit, 'clear' to reset memory)\n")

    while True:
        query = input("❓ Question: ")
        if query.lower() == "exit":
            break
        
        if query.lower() == "clear":
            memory.clear()
            print("\n🧹 Conversation memory cleared!\n")
            continue

        response = agent.run(query)
        print("\n💡 Answer:\n", response, "\n")


if __name__ == "__main__":
    main()
