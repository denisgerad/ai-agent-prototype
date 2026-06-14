#PDF + Tools + Agent + Debug Agent with Streamlit UI
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.tools import BaseTool, tool
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from pydantic import BaseModel, Field
import requests
import sys
import os

# Add agents directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))
from debug_agent import DebugAgent
from architect_agent import ArchitectAgent

# ---------- CONFIG ----------
PDF_PATH = "sample.pdf"
OLLAMA_MODEL = "mistral"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"
# ---------------------------




@st.cache_resource
def initialize_llm_and_embeddings():
    """Initialize just the LLM and embedding model (cached)"""
    llm = ChatOllama(model=OLLAMA_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return llm, embeddings


def build_pdf_tool(pdf_path: str, llm, embeddings):
    """Build a PDF search tool from a given PDF path"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    @tool
    def pdf_search(question: str) -> str:
        """Use this to answer questions about the uploaded PDF document. Input should be a clear question about the PDF content."""
        docs = retriever.invoke(question)
        context = "\n".join([d.page_content for d in docs])
        prompt = f"Based on this context:\n{context}\n\nAnswer this question: {question}"
        return llm.invoke(prompt)

    return pdf_search


def build_tools(llm, embeddings, pdf_path: str = None):
    """Build all tools, optionally including a PDF tool"""

    @tool
    def weather(city: str) -> str:
        """Get current weather for a city. Input should be a city name."""
        try:
            resp = requests.get(f"https://wttr.in/{city}?format=3", timeout=5)
            return resp.text
        except Exception:
            return f"Could not fetch weather for {city}"

    @tool
    def web_scraper(url: str) -> str:
        """Scrape text content from a webpage. Input should be a valid URL starting with http:// or https://"""
        try:
            web_loader = WebBaseLoader(url)
            docs = web_loader.load()
            return "\n".join([d.page_content[:1000] for d in docs])
        except Exception:
            return f"Could not scrape {url}"

    @tool
    def web_search(query: str) -> str:
        """Search the internet for general information, facts, news, or any topic."""
        try:
            search = DuckDuckGoSearchAPIWrapper()
            return search.run(query)
        except Exception as e:
            return f"Search error: {str(e)}"

    tools = [weather, web_scraper, web_search]

    if pdf_path and os.path.isfile(pdf_path):
        tools.insert(0, build_pdf_tool(pdf_path, llm, embeddings))

    return tools


def main():
    st.set_page_config(
        page_title="Multi-Agent RAG System",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Multi-Agent RAG System")
    st.markdown("Choose your agent: Generic (task execution), Debug (investigation), or Architect (system design)")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent_mode" not in st.session_state:
        st.session_state.agent_mode = "Generic Agent"
    
    if "architect_confidence_mode" not in st.session_state:
        st.session_state.architect_confidence_mode = False
    
    if "debug_agent" not in st.session_state:
        st.session_state.debug_agent = DebugAgent()
    
    if "architect_agent" not in st.session_state:
        llm, _ = initialize_llm_and_embeddings()
        st.session_state.architect_agent = ArchitectAgent(model=lambda prompt: llm.invoke(prompt))
    
    if "agent" not in st.session_state:
        with st.spinner("🔧 Initializing agent system..."):
            llm, embeddings = initialize_llm_and_embeddings()
            tools = build_tools(llm, embeddings)
            st.session_state.agent = create_react_agent(llm, tools)
            st.session_state.llm = llm
            st.session_state.embeddings = embeddings
        st.success("✅ Agent ready!")

    # Sidebar
    with st.sidebar:
        st.header("🛠️ Controls")
        
        # Agent Mode Selection (Radio Buttons)
        st.subheader("🤖 Agent Mode")
        agent_mode = st.radio(
            "Select Agent:",
            ["Generic Agent", "Debug Agent", "Architect Agent"],
            index=["Generic Agent", "Debug Agent", "Architect Agent"].index(st.session_state.agent_mode),
            help="Choose which agent to use for your queries"
        )
        
        if agent_mode != st.session_state.agent_mode:
            st.session_state.agent_mode = agent_mode
            st.rerun()
        
        # Confidence Mode Toggle (only for Architect Agent)
        if st.session_state.agent_mode == "Architect Agent":
            confidence_mode = st.checkbox(
                "🎯 Include Confidence & Assumptions",
                value=st.session_state.architect_confidence_mode,
                help="Add confidence scores and explicit assumptions to architecture analysis"
            )
            if confidence_mode != st.session_state.architect_confidence_mode:
                st.session_state.architect_confidence_mode = confidence_mode
        
        # Show current mode info
        if st.session_state.agent_mode == "Generic Agent":
            st.info("💬 Generic Agent - Task execution with tools")
        elif st.session_state.agent_mode == "Debug Agent":
            st.info("🐛 Debug Agent - Investigation-first reasoning")
        elif st.session_state.agent_mode == "Architect Agent":
            st.info("🏗️ Architect Agent - System architecture analysis")
        
        st.divider()
        st.subheader("📄 PDF Upload")
        uploaded_pdf = st.file_uploader("Upload a PDF to enable PDF search", type="pdf")
        if uploaded_pdf is not None:
            import tempfile
            tmp_path = os.path.join(tempfile.gettempdir(), uploaded_pdf.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            if st.session_state.get("pdf_path") != tmp_path:
                st.session_state.pdf_path = tmp_path
                # Rebuild tools with the new PDF and recreate agent
                with st.spinner("📄 Processing PDF..."):
                    tools = build_tools(st.session_state.llm, st.session_state.embeddings, tmp_path)
                    st.session_state.agent = create_react_agent(st.session_state.llm, tools)
                st.success(f"✅ PDF loaded: {uploaded_pdf.name}")
        
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.debug_agent.reset()
            st.rerun()
        
        # Show past architecture decisions
        if st.session_state.agent_mode == "Architect Agent":
            if st.session_state.architect_agent.architecture_memory:
                with st.expander("📚 Past Architecture Decisions"):
                    for idx, decision in enumerate(reversed(st.session_state.architect_agent.architecture_memory[-5:]), 1):
                        st.caption(f"**Decision {idx}:** {decision['request'][:80]}...")
        
        st.divider()
        
        st.header("📋 Available Capabilities")
        if st.session_state.agent_mode == "Generic Agent":
            pdf_status = "✅ PDF_Search: Query the PDF document" if st.session_state.get("pdf_path") else "⬆️ Upload a PDF to enable PDF search"
            st.markdown(f"""
            - {pdf_status}
            - **Weather**: Get weather for any city
            - **Web_Scraper**: Extract text from URLs
            - **Web_Search**: Search the internet
            """)
        elif st.session_state.agent_mode == "Debug Agent":
            st.markdown("""
            - **Investigation Mode**: Analyze bugs systematically
            - **Root Cause Analysis**: Score potential causes
            - **Verification Tests**: Generate test scenarios
            - **Fix Strategies**: Provide prioritized solutions
            """)
        elif st.session_state.agent_mode == "Architect Agent":
            st.markdown("""
            - **Architecture Analysis**: Evaluate system designs
            - **Trade-off Assessment**: Compare options
            - **Risk Identification**: Surface hidden issues
            - **Conditional Recommendations**: Context-aware advice
            """)
        
        st.divider()
        
        st.header("💡 Example Questions")
        if st.session_state.agent_mode == "Generic Agent":
            st.markdown("""
            - "What is the PDF about?"
            - "What's the weather in London?"
            - "Search for Python tutorials"
            - "Scrape content from a website"
            """)
        elif st.session_state.agent_mode == "Debug Agent":
            st.markdown("""
            - "My delete function works on PC but not on mobile"
            - "Getting 'no token' error on iPhone Safari"
            - "App crashes only sometimes on Android"
            - "Payment fails but only for some users"
            """)
        elif st.session_state.agent_mode == "Architect Agent":
            st.markdown("""
            - "I want to build a messaging app with user login using MERN stack"
            - "Help me design a multi-tenant SaaS architecture"
            - "What's the best way to implement real-time notifications?"
            - "I need to architect a fintech payment system"
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

        # Route to appropriate agent based on mode
        agent_mode = st.session_state.agent_mode

        # Get agent response
        with st.chat_message("assistant"):
            spinner_text = {
                "Generic Agent": "Thinking...",
                "Debug Agent": "🐛 Analyzing with Debug Agent...",
                "Architect Agent": "🏗️ Architecting solution..."
            }
            
            with st.spinner(spinner_text.get(agent_mode, "Thinking...")):
                try:
                    if agent_mode == "Debug Agent":
                        # Use debug agent
                        # Build conversation history from session messages
                        history = ""
                        for msg in st.session_state.messages[:-1]:  # exclude current prompt
                            role = "User" if msg["role"] == "user" else "Assistant"
                            history += f"{role}: {msg['content']}\n"
                        
                        # Handle with debug agent
                        result = st.session_state.debug_agent.handle(prompt, conversation_history=history)
                        
                        if result["mode"] == "INVESTIGATION":
                            response = result["formatted_response"]
                            st.markdown(response)
                        else:
                            # ANALYSIS mode - send to LLM
                            llm, _ = initialize_llm_and_embeddings()
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
                        
                        st.session_state.messages.append({"role": "assistant", "content": response if result["mode"] == "INVESTIGATION" else full_response})
                    
                    elif agent_mode == "Architect Agent":
                        # Use architect agent with confidence mode if enabled
                        response = st.session_state.architect_agent.analyze(
                            prompt, 
                            include_confidence=st.session_state.architect_confidence_mode
                        )
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    else:
                        # Generic agent flow (default)
                        from langchain_core.messages import HumanMessage
                        response = st.session_state.agent.invoke({"messages": [HumanMessage(content=prompt)]})
                        
                        # Extract response from agent output
                        response_text = response.get("messages", [])[-1].content if response.get("messages") else str(response)
                        
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
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
