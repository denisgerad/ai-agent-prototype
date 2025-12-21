# 🤖 RAG Agent with Tools

A Retrieval-Augmented Generation (RAG) application that combines PDF document analysis with multiple external tools using LangChain and Ollama. Available in both CLI and Streamlit web interface versions.

## 🌟 Features

- **PDF Question Answering**: Query and extract information from PDF documents using vector embeddings
- **Weather Information**: Get real-time weather data for any city
- **Web Scraping**: Extract text content from any webpage
- **Web Search**: Search the internet using DuckDuckGo (Streamlit version)
- **Conversation Memory**: Maintains context across multiple questions
- **Agent-based Architecture**: Intelligent tool selection based on user queries

## 🛠️ Tech Stack

- **LangChain**: Framework for building LLM applications
- **Ollama**: Local LLM inference (Mistral model)
- **ChromaDB**: Vector database for document embeddings
- **HuggingFace Embeddings**: Sentence transformers for text embeddings
- **Streamlit**: Web interface (optional)
- **PyPDF**: PDF document processing

## 📋 Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Mistral model installed in Ollama: `ollama pull mistral`

## 🚀 Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd rag_test
```

2. Install required packages:
```bash
pip install langchain langchain-community langchain-classic langchain-huggingface langchain-ollama
pip install chromadb sentence-transformers pypdf requests
pip install streamlit  # For web interface
pip install duckduckgo-search  # For web search (Streamlit version)
```

3. Ensure Ollama is running:
```bash
ollama serve
```

4. Place your PDF file as `sample.pdf` in the project directory

## 💻 Usage

### CLI Version

Run the command-line interface:
```bash
python app.py
```

Commands:
- Type your question and press Enter
- Type `clear` to reset conversation memory
- Type `exit` to quit

### Web Interface (Streamlit)

Launch the Streamlit app:
```bash
streamlit run app_streamlit.py
```

The web interface includes:
- Chat-style conversation interface
- Sidebar with available tools
- Example questions
- Clear conversation button

## 📝 Example Questions

- "What is this PDF about?"
- "Summarize the main points in the document"
- "What's the weather in London?"
- "Scrape https://example.com"
- "Search for the latest news about AI" (Streamlit version)
- Follow-up questions like "Can you elaborate on that?" (uses conversation memory)

## ⚙️ Configuration

Edit the configuration variables in either `app.py` or `app_streamlit.py`:

```python
PDF_PATH = "sample.pdf"              # Path to your PDF file
OLLAMA_MODEL = "mistral"             # Ollama model to use
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"           # Vector database storage
```

## 🏗️ Architecture

1. **Document Processing**:
   - PDF is loaded and split into chunks
   - Chunks are embedded using HuggingFace models
   - Embeddings are stored in ChromaDB

2. **Agent System**:
   - LangChain agent with CONVERSATIONAL_REACT_DESCRIPTION type
   - Multiple tools available for different tasks
   - Conversation memory for context retention

3. **Available Tools**:
   - `PDF_Search`: RAG-based document QA
   - `Weather`: Weather API integration
   - `Web_Scraper`: Webpage content extraction
   - `Web_Search`: Internet search (Streamlit only)

## 📁 Project Structure

```
rag_test/
├── app.py                  # CLI version
├── app_streamlit.py        # Web interface version
├── sample.pdf              # Your PDF document
├── chroma_db/              # Vector database storage
├── README.md               # This file

```

## 🔧 Troubleshooting

**Ollama connection error**:
- Ensure Ollama is running: `ollama serve`
- Verify Mistral model is installed: `ollama list`

**Embedding errors**:
- First run will download the embedding model (may take time)
- Ensure internet connection for initial model download

**Memory issues**:
- Use `clear` command (CLI) or Clear Conversation button (Streamlit)
- Restart the application

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements.

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the framework
- [Ollama](https://ollama.ai/) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Streamlit](https://streamlit.io/) for the web interface
