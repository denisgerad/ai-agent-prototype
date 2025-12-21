from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load PDFs from folder
documents = SimpleDirectoryReader("./").load_data()

# Create embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create index
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

# LLM
llm = Ollama(model="mistral")

# Query
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("Please explain WARRANTY POLICY")

print(response)