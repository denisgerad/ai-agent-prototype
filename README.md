AI Agent Prototype
Agentic Reasoning · Memory · Tool Use · Debugging Intelligence

A lightweight AI agent proof-of-concept demonstrating:

🧠 Conversation memory
🛠️ Dynamic tool usage
🔍 Investigation-first debugging
🤖 Agentic reasoning beyond code refactoring

This project intentionally focuses on how an AI agent reasons, asks questions, and decides what to inspect—not just on generating code.

The application is available as both a CLI tool and a Streamlit web interface.

🌟 Core Capabilities
🔹 Generic Agent (RAG + Tools)
PDF Question Answering (RAG)
Uses embeddings and a vector database to retrieve relevant PDF chunks.

Conversation Memory
Maintains conversational context across turns for follow-up questions.

Web Scraping
Extracts and reasons over content from arbitrary webpages.

Web Search (Streamlit only)
Performs live DuckDuckGo searches.

Weather Tool
Fetches real-time weather data for user-specified locations.

Agentic Tool Selection
ReAct-style reasoning determines when and which tool to invoke.

🔹 Debug Agent (Investigation-First AI)
A specialized AI Debug Agent designed to overcome the common failure mode of tools like Copilot:

❌ Blind code refinement without understanding the real-world test case

Instead, this agent:
Detects environmental and platform signals
Asks clarifying questions before refactoring
Separates storage, transport, and auth concerns
Produces testable, 60–90 second verification steps
This turns the AI into a debugging co-pilot, not a theorist.

🧠 Debug Agent Design Philosophy
Key Principle:

If something works in one environment, the core logic is likely correct.

The agent prioritizes:
Platform differences (PC vs Mobile, Chrome vs Safari)
Browser storage limitations
Event-handling differences (click vs touch)
Authentication and transport failures
Only after verification does it suggest fixes.

🛠️ Tech Stack
LangChain – Agent & tool orchestration
Ollama – Local LLM inference (Mistral)
ChromaDB – Vector database for RAG
HuggingFace Embeddings – MiniLM sentence transformers
Streamlit – Web-based chat UI
PyPDF – PDF document parsing

📋 Prerequisites
Python 3.8+
Ollama installed and running
Mistral model pulled:
ollama pull mistral

🚀 Installation
git clone <your-repo-url>
cd <repo-folder>

pip install langchain langchain-community langchain-classic \
            langchain-huggingface langchain-ollama
pip install chromadb sentence-transformers pypdf requests
pip install streamlit duckduckgo-search


Start Ollama:
ollama serve

Add a PDF:
sample.pdf

💻 Usage
CLI Mode
python app.py

Commands:
Ask questions directly
clear → Reset conversation memory
exit → Quit
debug mode → Enable Debug Agent

Streamlit Web UI (Recommended)
streamlit run app_streamlit.py

Features:
Chat-style UI
Debug Mode toggle
Tool visibility
Stuctured debug output
Clear conversation button

🧪 Debug Agent Workflow
User Flow:
User describes issue
“Delete works on PC but not on mobile”

Investigation Mode Activated
Detects environmental signals
Asks clarifying questions

Analysis Mode
Root cause confidence scoring
Platform-aware hypotheses

Verification
60–90 second test steps
Concrete console commands

Inspection
Exact files and headers to check

Fix Strategy
Multiple options
No forced refactor

📊 Example Debug Output (Summarized)
Root cause likelihood with confidence scores
iOS Safari–specific verification steps
Axios interceptor inspection checklist
Token storage vs transport separation

Fix strategies:
httpOnly cookies (recommended)
sessionStorage fallback
IndexedDB backup

📂 Project Structure
.
├── app.py                      # CLI agent
├── app_streamlit.py            # Streamlit UI
├── sample.pdf                  # Example document
├── chroma_db/                  # Vector storage
├── agents/
│   ├── code_inspector.py
│   ├── verification_generator.py
│   ├── root_cause_analyzer.py
│   ├── fix_strategy_generator.py
├── tests/
│   ├── test_debug_agent.py
│   ├── test_error_priority.py
│   └── test_enhanced_features.py
├── README.md

🧪 Testing
python test_debug_agent.py
python test_error_priority.py
python test_enhanced_features.py

🔧 Troubleshooting
Ollama
Ensure running: ollama serve
Verify model: ollama list

Embeddings
First run downloads MiniLM model
Ensure internet access

Memory
Use clear or “Clear Conversation”
Restart app if needed

🎯 Design Scope
This project is not a production RAG system.
It is a:
Research prototype
Debugging intelligence experiment
Hands-on exploration of agentic AI patterns

🤝 Contributing
Issues, discussions, and improvements are welcome.

📄 License
MIT License

🙏 Acknowledgments
LangChain – Agent & tool abstractions
Ollama – Local LLM inference
ChromaDB – Vector storage
Streamlit – Rapid UI development

🚀 Final Note
The Debug Agent does not guess.
It investigates, verifies, and only then fixes.