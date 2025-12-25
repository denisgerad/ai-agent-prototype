#PDF + Tools + Agent + Debug Agent with Streamlit UI
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_classic.tools import Tool
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
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
    try:
        resp = requests.get(f"https://wttr.in/{city}?format=3", timeout=5)
        return resp.text
    except:
        return f"Could not fetch weather for {city}"

def scrape_website(url: str) -> str:
    """Scrape text content from a webpage"""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return "\n".join([d.page_content[:1000] for d in docs])
    except:
        return f"Could not scrape {url}"

def web_search(query: str) -> str:
    """Search the web for general information"""
    try:
        search = DuckDuckGoSearchAPIWrapper()
        results = search.run(query)
        return results
    except Exception as e:
        return f"Search error: {str(e)}"
# -------------------------------


@st.cache_resource
def initialize_agent_system():
    """Initialize the RAG system and agent (cached)"""
    
    # Load PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # Create vector store
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    # Initialize LLM
    llm = OllamaLLM(model=OLLAMA_MODEL)

    # PDF QA
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=False
    )

    # Tools
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

    web_search_tool = Tool(
        name="Web_Search",
        func=web_search,
        description="Search the internet for general information, facts, news, or any topic. Use this for questions that require current information or general knowledge from the web. IMPORTANT: When using this tool, include relevant context from the conversation history in your search query to get accurate results."
    )

    tools = [pdf_tool, weather_tool, web_scraper_tool, web_search_tool]

    return llm, tools


def main():
    st.set_page_config(
        page_title="RAG Agent Chat",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 RAG Agent with Tools & Debug Mode")
    st.markdown("Chat with your PDF, get weather, scrape websites, and get intelligent debugging help!")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    
    if "debug_agent" not in st.session_state:
        st.session_state.debug_agent = DebugAgent()
    
    if "debug_memory" not in st.session_state:
        st.session_state.debug_memory = ConversationBufferMemory(
            memory_key="debug_history",
            return_messages=True,
            output_key="output"
        )
    
    if "agent" not in st.session_state:
        with st.spinner("🔧 Initializing agent system..."):
            llm, tools = initialize_agent_system()
            
            # Create memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
            
            # Create agent
            st.session_state.agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=10,
                max_execution_time=60,
                early_stopping_method="generate",
                memory=memory,
                agent_kwargs={
                    "prefix": """Assistant is a helpful AI that can use tools to answer questions. When the user asks a follow-up question, always consider the previous conversation context to understand what they're referring to. If you need to search for information, include the relevant topic from the conversation history in your search query.

For example:
- If the conversation is about "one tap trading" and the user asks "list popular apps", search for "popular one tap trading apps".
- If discussing a city and asked "what's the temperature", use that city name.

Always provide the most relevant and contextual answer based on the full conversation."""
                }
            )
            st.session_state.memory = memory
        st.success("✅ Agent ready!")

    # Sidebar
    with st.sidebar:
        st.header("🛠️ Controls")
        
        # Debug Mode Toggle
        debug_mode = st.toggle("🐛 Debug Mode", value=st.session_state.debug_mode)
        if debug_mode != st.session_state.debug_mode:
            st.session_state.debug_mode = debug_mode
            st.rerun()
        
        if st.session_state.debug_mode:
            st.info("🐛 Debug mode is ON - Bug reports will be intelligently analyzed")
        
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.session_state.debug_memory.clear()
            st.session_state.debug_agent.reset()
            st.rerun()
        
        st.divider()
        
        st.header("📋 Available Tools")
        st.markdown("""
        - **PDF_Search**: Query the PDF document
        - **Weather**: Get weather for any city
        - **Web_Scraper**: Extract text from URLs
        - **Web_Search**: Search the internet (like Google)
        - **🐛 Debug Agent**: Intelligent bug analysis
        """)
        
        st.divider()
        
        st.header("💡 Example Questions")
        st.markdown("""
        **Normal Mode:**
        - "What is the PDF about?"
        - "What's the weather in London?"
        - "Search for Python tutorials"
        
        **Debug Mode:**
        - "My delete function works on PC but not on mobile"
        - "Getting 'no token' error on iPhone Safari"
        - "App crashes only sometimes on Android"
        """)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check if should use debug agent
        # First check if debug mode is enabled
        # Then check for debug keywords OR environmental signals
        use_debug = st.session_state.debug_mode
        
        if not use_debug:
            # Check for explicit debug keywords
            use_debug = st.session_state.debug_agent.is_debug_query(prompt)
        
        if not use_debug:
            # Check for environmental signals even without explicit bug keywords
            from signal_detector import detect_signals, should_investigate
            signals = detect_signals(prompt)
            use_debug = should_investigate(signals)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..." if not use_debug else "🐛 Analyzing with Debug Agent..."):
                try:
                    if use_debug:
                        # Extract conversation history from debug memory
                        history = ""
                        if hasattr(st.session_state.debug_memory, 'chat_memory') and hasattr(st.session_state.debug_memory.chat_memory, 'messages'):
                            for msg in st.session_state.debug_memory.chat_memory.messages:
                                if hasattr(msg, 'type'):
                                    if msg.type == 'human':
                                        history += f"User: {msg.content}\n"
                                    elif msg.type == 'ai':
                                        history += f"Assistant: {msg.content}\n"
                        
                        # Handle with debug agent
                        result = st.session_state.debug_agent.handle(prompt, conversation_history=history)
                        
                        if result["mode"] == "INVESTIGATION":
                            response = result["formatted_response"]
                            st.markdown(response)
                            # Store in debug memory
                            st.session_state.debug_memory.save_context(
                                {"input": prompt},
                                {"output": response}
                            )
                        else:
                            # ANALYSIS mode - send to LLM
                            llm, _ = initialize_agent_system()
                            response = llm.invoke(result["prompt"])
                            
                            # Display the LLM analysis first
                            st.markdown("### 🤖 AI Analysis")
                            st.markdown(response)
                            
                            # Display confirmation gate BEFORE fix strategies
                            if result.get("confirmation_gate"):
                                st.markdown("---")
                                st.warning(result["confirmation_gate"])
                            
                            # Display additional analysis components
                            if result.get("root_cause_scores"):
                                st.markdown("---")
                                st.markdown(result["root_cause_scores"])
                            
                            if result.get("verification_tests"):
                                st.markdown("---")
                                st.markdown(result["verification_tests"])
                            
                            if result.get("inspection_checklist"):
                                st.markdown("---")
                                st.markdown(result["inspection_checklist"])
                            
                            # Show fix strategies only if no confirmation gate or after confirmation
                            if result.get("fix_strategies") and not result.get("confirmation_gate"):
                                st.markdown("---")
                                st.markdown(result["fix_strategies"])
                            elif result.get("fix_strategies") and result.get("confirmation_gate"):
                                st.info("💡 Fix strategies will be provided after you confirm the exact issue above.")
                            
                            # Combine all for storage
                            full_response = response
                            if result.get("confirmation_gate"):
                                full_response += "\n\n" + result["confirmation_gate"]
                            if result.get("root_cause_scores"):
                                full_response += "\n\n" + result["root_cause_scores"]
                            if result.get("verification_tests"):
                                full_response += "\n\n" + result["verification_tests"]
                            if result.get("inspection_checklist"):
                                full_response += "\n\n" + result["inspection_checklist"]
                            if result.get("fix_strategies") and not result.get("confirmation_gate"):
                                full_response += "\n\n" + result["fix_strategies"]
                            
                            # Store in debug memory
                            st.session_state.debug_memory.save_context(
                                {"input": prompt},
                                {"output": full_response}
                            )
                        
                        st.session_state.messages.append({"role": "assistant", "content": response if result["mode"] == "INVESTIGATION" else full_response})
                    else:
                        # Normal agent flow
                        response = st.session_state.agent.run(prompt)
                        
                        # Clean up any error messages in the response
                        if "For troubleshooting, visit:" in response:
                            response = response.split("For troubleshooting, visit:")[0].strip()
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_str = str(e)
                    
                    # Extract actual content from parsing errors
                    if "Could not parse LLM output:" in error_str:
                        # Extract the actual response between "output:" and "For troubleshooting"
                        content = error_str.split("Could not parse LLM output:")[1]
                        if "For troubleshooting, visit:" in content:
                            content = content.split("For troubleshooting, visit:")[0]
                        content = content.strip()
                        st.markdown(content)
                        st.session_state.messages.append({"role": "assistant", "content": content})
                    else:
                        error_msg = f"Error: {error_str}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
