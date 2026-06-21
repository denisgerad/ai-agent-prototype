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
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== INTERCEPT HTTPX TO LOG API REQUESTS =====
import httpx
original_post = httpx.Client.post
original_post_async = httpx.AsyncClient.post

def log_post(self, *args, **kwargs):
    """Log httpx POST requests"""
    try:
        if 'ollama' in str(args) or '11434' in str(args):
            logger.info(f"[OLLAMA] POST request to: {args[0] if args else 'unknown'}")
            if 'json' in kwargs and kwargs['json']:
                json_payload = kwargs['json']
                logger.info(f"[OLLAMA] Request body: {json.dumps(json_payload, indent=2, default=str)[:1000]}")
    except Exception as e:
        logger.debug(f"[OLLAMA] Logging error: {e}")
    return original_post(self, *args, **kwargs)

async def log_post_async(self, *args, **kwargs):
    """Log async httpx POST requests"""
    try:
        if 'ollama' in str(args) or '11434' in str(args):
            logger.info(f"[OLLAMA_ASYNC] POST request to: {args[0] if args else 'unknown'}")
            if 'json' in kwargs and kwargs['json']:
                json_payload = kwargs['json']
                logger.info(f"[OLLAMA_ASYNC] Request body: {json.dumps(json_payload, indent=2, default=str)[:1000]}")
    except Exception as e:
        logger.debug(f"[OLLAMA_ASYNC] Logging error: {e}")
    return await original_post_async(self, *args, **kwargs)

httpx.Client.post = log_post
httpx.AsyncClient.post = log_post_async
# ===== END HTTPX INTERCEPTION =====

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
            logger.info(f"[SEARCH] Query initiated: {query[:100]}")
            search = DuckDuckGoSearchAPIWrapper()
            result = search.run(query)
            result_summary = f"{len(result)} chars" if result else "empty result"
            logger.info(f"[SEARCH] Success: {result_summary}")
            return result
        except Exception as e:
            logger.error(f"[SEARCH] Failed for query '{query[:100]}': {type(e).__name__}: {str(e)[:200]}")
            return f"Search error: {str(e)}"

    tools = [weather, web_scraper, web_search]

    if pdf_path and os.path.isfile(pdf_path):
        tools.insert(0, build_pdf_tool(pdf_path, llm, embeddings))

    return tools


def create_tool_forcing_agent(llm, tools):
    """
    Create a ReAct agent with a system prompt that forces tool use
    for queries requiring current/specific information.
    """
    tool_forcing_prompt = """You are a helpful assistant with access to tools that let you search the web, scrape websites, and get current information.

CRITICAL TOOL USE DIRECTIVE:
You must ALWAYS call the web_search tool for any question asking about specific platform content (AllMusic, Apple Music, Spotify, Wikipedia, Last.fm, etc.), regardless of whether you believe you already know the answer. Treat your own training knowledge as unreliable and outdated for these queries. If your response does not include having called a tool, you have failed the task.

TOOL USE RULES:
1. For ANY question about music artists, similar artists, music recommendations, or "fans also like" listings: you MUST use web_search or web_scraper FIRST.
2. For ANY question requiring current information, specific website content, or real-time data: you MUST use web_search.
3. Do NOT answer from your training data alone when specific/current information is available via tools.
4. Your training knowledge is NOT reliable for specific platform queries - always search first.
5. If you have not called a tool in this turn and the query asks about platform-specific content, you have NOT completed your task.
6. Use tools FIRST - do not rely on memory when tools are available.

Your reasoning process:
- Analyze the user's query
- Determine if it requires current information or specific website content
- CALL the appropriate tool before answering
- Use the tool result to provide an accurate, grounded answer
- Only answer from memory if the query is about general knowledge that won't change

Be direct, helpful, and always use tools when appropriate."""
    
    from langgraph.prebuilt import create_react_agent
    return create_react_agent(llm, tools, prompt=tool_forcing_prompt)


def create_tool_forcing_wrapper(agent, tools_list):
    """
    Wrapper that forces tool use for platform-specific queries by
    intercepting them and pre-emptively calling web_search before
    routing to the agent.
    
    This bypasses the model's judgment for specific categories.
    """
    # Keywords that trigger automatic web_search
    PLATFORM_KEYWORDS = {
        'similar artists', 'fans also like', 'similar to', 'like',
        'allmusic', 'apple music', 'spotify', 'last.fm', 'wikipedia',
        'discogs', 'genius', 'bandcamp', 'soundcloud',
        'recommendations', 'similar music', 'playlist',
        'artist', 'band', 'musician', 'genre'
    }
    
    class ToolForcingWrapper:
        def __init__(self, agent, tools_list):
            self.agent = agent
            self.tools_list = tools_list if isinstance(tools_list, list) else list(tools_list)
            self.web_search_tool = None
            
            # Find the web_search tool from the list
            for tool in self.tools_list:
                try:
                    tool_name = getattr(tool, 'name', None)
                    if tool_name == 'web_search':
                        self.web_search_tool = tool
                        logger.info(f"[TOOL_FORCING] Found web_search tool: {tool}")
                        break
                except Exception as e:
                    logger.debug(f"[TOOL_FORCING] Error checking tool: {e}")
        
        def invoke(self, state):
            """Intercept agent invocation and force tool use if needed"""
            from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
            
            messages = state.get("messages", [])
            
            # Get the last user message
            user_message = None
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    user_message = msg.content
                    break
            
            if user_message:
                query_lower = user_message.lower()
                
                # Check if query contains platform keywords
                needs_search = any(keyword in query_lower for keyword in PLATFORM_KEYWORDS)
                
                if needs_search:
                    logger.info(f"[TOOL_FORCING] Query needs platform search: '{query_lower[:80]}'")
                    
                    # Pre-emptively call web_search with focused, platform-specific queries
                    if self.web_search_tool:
                        try:
                            import re
                            
                            # Extract artist name from the user message
                            # Look for patterns like "Search for X's similar artists" or "similar artists [for/of] X"
                            artist_name = None
                            
                            # Try to extract from "search for X's" or "search for X and"
                            match = re.search(r'search\s+for\s+([^\'"\n]+?)(?:\'s|\'|\s+(?:and|or|on|similar))', user_message, re.IGNORECASE)
                            if match:
                                artist_name = match.group(1).strip()
                            
                            # Fallback: look for quoted names
                            if not artist_name:
                                match = re.search(r'"([^"]+)"', user_message)
                                if match:
                                    artist_name = match.group(1).strip()
                            
                            # Final fallback: just grab first few meaningful words
                            if not artist_name:
                                words = [w for w in user_message.split() if len(w) > 2 and w.lower() not in ['search', 'for', 'the', 'and', 'from', 'about']]
                                if words:
                                    artist_name = ' '.join(words[:2])
                            
                            if not artist_name:
                                artist_name = "artist"
                            
                            logger.info(f"[TOOL_FORCING] Extracted artist name: '{artist_name}'")
                            
                            # Create three focused platform-specific queries
                            platform_queries = [
                                f"{artist_name} similar artists AllMusic",
                                f"{artist_name} fans also like Apple Music", 
                                f"{artist_name} similar artists Spotify"
                            ]
                            
                            # Execute searches and combine results
                            all_results = []
                            for platform_query in platform_queries:
                                logger.info(f"[TOOL_FORCING] Executing focused query: '{platform_query}'")
                                try:
                                    result = self.web_search_tool.invoke(platform_query)
                                    all_results.append(result)
                                    logger.info(f"[TOOL_FORCING] Platform query result: {len(result)} chars")
                                except Exception as e:
                                    logger.error(f"[TOOL_FORCING] Platform query failed: {e}")
                            
                            # Combine all results
                            search_result = "\n---\n".join(all_results) if all_results else "No results found"
                            logger.info(f"[TOOL_FORCING] Combined search result: {len(search_result)} chars")
                            
                            # Inject the search result as a tool message in the state
                            # This simulates the tool having been called
                            tool_msg = ToolMessage(
                                content=search_result,
                                tool_call_id="forced_web_search",
                                name="web_search"
                            )
                            
                            # Add a fake agent message that "called" the tool
                            from langchain_core.messages.tool import ToolCall
                            agent_msg = AIMessage(
                                content="",
                                tool_calls=[ToolCall(id="forced_web_search", name="web_search", args={"query": f"similar artists for {artist_name}"})]
                            )
                            
                            # Insert these before the final agent call
                            state["messages"] = messages + [agent_msg, tool_msg]
                            logger.info(f"[TOOL_FORCING] Injected pre-search into message state")
                        except Exception as e:
                            logger.error(f"[TOOL_FORCING] Pre-search failed: {type(e).__name__}: {e}")
                    else:
                        logger.warning("[TOOL_FORCING] web_search tool not found in tools list")
            
            # Now invoke the actual agent
            logger.info(f"[TOOL_FORCING] Invoking agent with {len(state.get('messages', []))} messages")
            result = self.agent.invoke(state)
            return result
    
    return ToolForcingWrapper(agent, tools_list)


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
            logger.info(f"[AGENT_INIT] Building agent with {len(tools)} tools: {[t.name if hasattr(t, 'name') else str(t) for t in tools]}")
            base_agent = create_tool_forcing_agent(llm, tools)
            # Wrap with tool-forcing interceptor
            st.session_state.agent = create_tool_forcing_wrapper(base_agent, tools)
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
                    base_agent = create_tool_forcing_agent(st.session_state.llm, tools)
                    st.session_state.agent = create_tool_forcing_wrapper(base_agent, tools)
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
                        
                        logger.info(f"[AGENT] Generic Agent invoked with query: {prompt[:100]}")
                        
                        # Log agent graph structure
                        try:
                            agent_graph = st.session_state.agent
                            if hasattr(agent_graph, 'nodes'):
                                logger.info(f"[AGENT] Agent graph nodes: {list(agent_graph.nodes.keys())}")
                            if hasattr(agent_graph, 'get_graph'):
                                logger.info(f"[AGENT] Agent has get_graph method")
                        except Exception as e:
                            logger.debug(f"[AGENT] Could not inspect agent graph: {e}")
                        
                        response = st.session_state.agent.invoke({"messages": [HumanMessage(content=prompt)]})
                        
                        # Log the full response to understand reasoning
                        messages = response.get("messages", [])
                        logger.info(f"[AGENT] Response contains {len(messages)} messages")
                        for i, msg in enumerate(messages):
                            msg_type = type(msg).__name__
                            logger.info(f"[AGENT] Message {i}: {msg_type}")
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                tool_names = []
                                for tc in msg.tool_calls:
                                    tool_names.append(getattr(tc, 'name', getattr(tc, 'tool', 'unknown')))
                                logger.info(f"[AGENT]   -> Tool calls: {tool_names}")
                            if hasattr(msg, 'content'):
                                content_preview = str(msg.content)[:150] if msg.content else "(empty)"
                                logger.info(f"[AGENT]   -> Content: {content_preview}")
                        
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
