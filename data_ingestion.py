import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import Client, Settings
from chromadb.utils import embedding_functions

# Configure ChromaDB Client (Persistent Storage)
CHROMA_DATA_PATH = "./chroma_db"
COLLECTION_NAME = "financial_reports"

def get_chroma_collection():
    """Initializes and returns the ChromaDB collection."""
    # Using the default sentence-transformers model from ChromaDB
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    client = Client(Settings(persist_directory=CHROMA_DATA_PATH, is_persistent=True))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def process_pdf_and_store(file_path: str):
    """
    Parses a PDF or TXT, chunks it, and stores the embeddings into ChromaDB.
    """
    print(f"Loading {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Load File
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".txt"):
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError("Unsupported file format. Only PDF and TXT are supported.")
        
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Store in ChromaDB
    collection = get_chroma_collection()
    
    # Prepare data for Chroma
    documents_list = [chunk.page_content for chunk in chunks]
    metadatas_list = [{"source": file_path, "page": chunk.metadata.get("page", 0)} for chunk in chunks]
    ids_list = [f"{os.path.basename(file_path)}_chunk_{i}" for i in range(len(chunks))]

    print("Generating embeddings and saving to ChromaDB...")
    collection.add(
        documents=documents_list,
        metadatas=metadatas_list,
        ids=ids_list
    )
    print("Ingestion complete!")

def process_directory(directory_path: str):
    """
    Parses all supported files (PDF, TXT) in the given directory.
    """
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return

    for filename in os.listdir(directory_path):
        if filename.lower().endswith((".pdf", ".txt")):
            file_path = os.path.join(directory_path, filename)
            try:
                process_pdf_and_store(file_path)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    # Example usage: python data_ingestion.py report.pdf OR python data_ingestion.py reports/
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isdir(path):
            print(f"Processing directory: {path}")
            process_directory(path)
        else:
            process_pdf_and_store(path)
    else:
        print("Usage: python data_ingestion.py <file_or_directory_path>")
