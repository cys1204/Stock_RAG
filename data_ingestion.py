import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import Client, Settings
from chromadb.utils import embedding_functions
import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
def extract_document_metadata(text: str) -> dict:
    """Uses LLM to extract company name and year from the beginning of the document."""
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        system_prompt = (
            "你是一個資料分析助手。請從以下財報內容的開頭擷取「公司全名或簡稱」與「財報年份」。\n"
            "請只輸出合法的 JSON 格式，不要包含任何其他文字或 Markdown 標籤，例如：{\"company\": \"台積電\", \"year\": \"114\"}。"
        )
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"文件內容：\n{text[:2000]}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        data = json.loads(result)
        return {"company": data.get("company", "Unknown"), "year": str(data.get("year", "Unknown"))}
    except Exception as e:
        print(f"Metadata extraction failed: {e}")
        return {"company": "Unknown", "year": "Unknown"}

# Configure ChromaDB Client (Persistent Storage)
CHROMA_DATA_PATH = "./chroma_db"
COLLECTION_NAME = "financial_reports"

def get_chroma_collection():
    """Initializes and returns the ChromaDB collection."""
    # Using a multilingual model that understands Chinese properly
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    import chromadb
    client = chromadb.PersistentClient(
        path=CHROMA_DATA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
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
        import pymupdf4llm
        from langchain.docstore.document import Document
        
        # Convert to Markdown
        md_text = pymupdf4llm.to_markdown(file_path)
        
        # Save to markdown_cache for inspection
        os.makedirs("markdown_cache", exist_ok=True)
        base_name = os.path.basename(file_path).replace(".pdf", ".md")
        md_path = os.path.join("markdown_cache", base_name)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)
            
        documents = [Document(page_content=md_text, metadata={"source": file_path, "page": 0})]
        print(f"Loaded {file_path} and converted to Markdown (saved to {md_path}).")
        
    elif file_path.lower().endswith((".txt", ".md")):
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        print(f"Loaded {len(documents)} txt/md pages.")
    else:
        raise ValueError("Unsupported file format. Only PDF, TXT, and MD are supported.")
        
    print("Extracting metadata via LLM...")
    doc_meta = extract_document_metadata(documents[0].page_content)
    print(f"Extracted Metadata: {doc_meta}")
    # 2. Split Text
    from langchain.text_splitter import MarkdownHeaderTextSplitter
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )

    chunks = []
    for doc in documents:
        # First split by Markdown Headers
        md_header_splits = markdown_splitter.split_text(doc.page_content)
        
        # Preserve original metadata (like source file)
        for split in md_header_splits:
            split.metadata.update(doc.metadata)
            
        # Then split by Character length if still too long
        sub_chunks = text_splitter.split_documents(md_header_splits)
        chunks.extend(sub_chunks)

    print(f"Split into {len(chunks)} chunks.")

    # 3. Store in ChromaDB
    collection = get_chroma_collection()
    
    # Prepare data for Chroma
    documents_list = [chunk.page_content for chunk in chunks]
    
    metadatas_list = []
    for chunk in chunks:
        meta = {
            "source": file_path, 
            "page": chunk.metadata.get("page", 0),
            "company": doc_meta["company"],
            "year": doc_meta["year"]
        }
        # Include markdown headers if present
        for key in ["Header 1", "Header 2", "Header 3"]:
            if key in chunk.metadata:
                meta[key] = chunk.metadata[key]
        metadatas_list.append(meta)
        
    ids_list = [f"{os.path.basename(file_path)}_chunk_{i}" for i in range(len(chunks))]

    print("Generating embeddings and saving to ChromaDB...")
    collection.upsert(
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
        if filename.lower().endswith((".pdf", ".txt", ".md")):
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
