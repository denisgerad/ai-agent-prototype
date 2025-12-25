#PDF + Tools + Agent + Debug Agent
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_classic.tools import Tool
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.memory import ConversationBufferMemory
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
    
    # ---------- DEBUG MEMORY ----------
    debug_memory = ConversationBufferMemory(
        memory_key="debug_history",
        return_messages=True,
        output_key="output"
    )
    
    # ---------- DEBUG AGENT ----------
    debug_agent = DebugAgent()

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

    print("\n[+] Agent ready with conversation memory! Ask anything (type 'exit' to quit, 'clear' to reset memory, 'debug mode' to enter debug mode)\n")

    debug_mode = False
    
    while True:
        mode_indicator = "[DEBUG]" if debug_mode else "[?]"
        query = input(f"{mode_indicator} Question: ")
        
        if query.lower() == "exit":
            break
        
        if query.lower() == "clear":
            memory.clear()
            debug_memory.clear()
            debug_agent.reset()
            print("\n[*] Conversation memory cleared!\n")
            continue
        
        if query.lower() == "debug mode":
            debug_mode = not debug_mode
            status = "ON" if debug_mode else "OFF"
            print(f"\n[DEBUG] Debug mode {status}\n")
            continue
        
        # Route to debug agent if debug keywords detected or in debug mode
        if debug_mode or debug_agent.is_debug_query(query):
            print("\n[*] Routing to Debug Agent...\n")
            
            # Get conversation history from debug memory
            history = ""
            if hasattr(debug_memory, 'chat_memory') and hasattr(debug_memory.chat_memory, 'messages'):
                for msg in debug_memory.chat_memory.messages:
                    if hasattr(msg, 'type'):
                        if msg.type == 'human':
                            history += f"User: {msg.content}\n"
                        elif msg.type == 'ai':
                            history += f"Assistant: {msg.content}\n"
            
            # Handle with debug agent
            result = debug_agent.handle(query, conversation_history=history)
            
            if result["mode"] == "INVESTIGATION":
                print("\n" + result["formatted_response"] + "\n")
                # Store in debug memory
                debug_memory.save_context(
                    {"input": query},
                    {"output": result["formatted_response"]}
                )
            else:
                # ANALYSIS mode - send prompt to LLM
                llm_response = llm.invoke(result["prompt"])
                print("\n[*] Debug Analysis:\n", llm_response, "\n")
                # Store in debug memory
                debug_memory.save_context(
                    {"input": query},
                    {"output": llm_response}
                )
        else:
            # Normal agent flow
            response = agent.run(query)
            print("\n[*] Answer:\n", response, "\n")


if __name__ == "__main__":
    main()
