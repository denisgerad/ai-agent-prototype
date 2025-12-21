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
