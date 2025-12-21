# AI Agent Prototype with Memory and Tool Use

A lightweight **AI agent proof-of-concept** designed to demonstrate
**conversation memory, agentic reasoning, and dynamic tool usage**
using a locally hosted LLM.

Rather than focusing on model performance or UI polish, this project
explores how an AI agent **decides when and how to use different tools**
— including PDF-based retrieval (RAG), web scraping, web search, and
external APIs — while maintaining conversational context.

The application is available in both **CLI** and **Streamlit web interface**
forms.

---

## 🌟 Agent Capabilities

- **PDF Question Answering (RAG)**  
  Uses vector embeddings to retrieve relevant information from PDF documents
  as one of several tools available to the agent.

- **Conversation Memory**  
  Maintains conversational state across multiple turns, enabling follow-up
  questions and contextual reasoning.

- **Web Scraping**  
  Extracts and reasons over text content from arbitrary webpages.

- **Web Search**  
  Performs live internet searches using DuckDuckGo (Streamlit version).

- **Weather Information**  
  Retrieves real-time weather data for user-specified locations.

- **Agent-Based Architecture**  
  The agent dynamically selects tools based on user intent using a
  ReAct-style reasoning loop.

---

## 🛠️ Tech Stack

- **LangChain** – Framework for building agentic LLM applications  
- **Ollama** – Local LLM inference (Mistral model)  
- **ChromaDB** – Vector database for document embeddings  
- **HuggingFace Embeddings** – Sentence-transformer models for text embeddings  
- **Streamlit** – Optional web-based chat interface  
- **PyPDF** – PDF document processing  

---

## 📋 Prerequisites

- Python 3.8+
- Ollama installed and running
- Mistral model installed in Ollama:
  ```bash
  ollama pull mistral

🚀 Installation
Clone the repository

git clone <your-repo-url>
cd <repo-folder>
Install dependencies

pip install langchain langchain-community langchain-classic \
            langchain-huggingface langchain-ollama
pip install chromadb sentence-transformers pypdf requests
pip install streamlit                 # Web interface
pip install duckduckgo-search          # Web search (Streamlit)
Start Ollama

ollama serve
Add a PDF
Place your document as sample.pdf in the project directory.

💻 Usage
CLI Version
Run the command-line agent:
python app.py

Commands
Ask any question directly
clear → Reset conversation memory
exit → Quit the application

Web Interface (Streamlit)
Launch the web app:
streamlit run app_streamlit.py

Web UI features
Chat-style interaction
Sidebar showing available tools
Example prompts
Clear conversation button

📝 Example Questions
“What is this PDF about?”
“Summarize the key points in the document”
“Can you elaborate on that?” (uses memory)
“What’s the weather in London?”
“Scrape https://example.com”
“Search for the latest news on AI” (Streamlit)

⚙️ Configuration
Edit variables in app.py or app_streamlit.py:

PDF_PATH = "sample.pdf"
OLLAMA_MODEL = "mistral"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"

🏗️ Architecture Overview
1. Document Processing
PDF is loaded and split into chunks
Chunks are embedded using HuggingFace models
Embeddings are stored in ChromaDB

2. Agent System
LangChain conversational ReAct agent
Multiple tools exposed to the agent
Conversation memory for context retention

3. Available Tools
PDF_Search – RAG-based document retrieval
Web_Scraper – Webpage text extraction
Web_Search – Live internet search (Streamlit only)
Weather – External API integration

🧠 Design Focus
This project is intentionally scoped as a prototype.
The primary focus is on:
Agent reasoning and tool selection
Context-aware interactions
Exploring agentic AI patterns

It is not intended as a production-ready RAG system, but as a hands-on
exploration of how LLM-based agents interact with multiple data sources.

📁 Project Structure
.
├── app.py                  # CLI version
├── app_streamlit.py        # Streamlit web interface
├── sample.pdf              # Example PDF document
├── chroma_db/              # Vector database storage
├── README.md

🔧 Troubleshooting
Ollama connection issues
Ensure Ollama is running: ollama serve
Verify the model is installed: ollama list
Embedding errors
First run downloads the embedding model (may take time)
Ensure internet access during initial setup
Memory issues
Use clear (CLI) or “Clear Conversation” (UI)
Restart the application if needed

🤝 Contributing
Contributions, issues, and suggestions are welcome.

📄 License
MIT License

🙏 Acknowledgments
LangChain for agent and tool abstractions
Ollama for local LLM inference
ChromaDB for vector storage
Streamlit for rapid UI development