import os
from data_ingestion import get_chroma_collection

def retrieve_context(query: str, top_k: int = 4) -> str:
    """
    Retrieves the most relevant chunks for the given query from ChromaDB.
    Returns the concatenated text chunks with their sources.
    """
    try:
        collection = get_chroma_collection()
    except Exception as e:
        print(f"Error accessing ChromaDB: {e}")
        return ""

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    if not results['documents'] or not results['documents'][0]:
        return ""

    context_parts = []
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    for i, doc in enumerate(documents):
        source = metadatas[i].get("source", "Unknown")
        # Ensure we only have the filename
        source_name = os.path.basename(source)
        page = metadatas[i].get("page", 0)
        
        context_parts.append(f"[{source_name}_Page_{page}]\n{doc}")

    return "\n\n".join(context_parts)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = sys.argv[1]
        context = retrieve_context(query)
        print("Retrieved Context:\n")
        print(context)
    else:
        print("Usage: python retriever.py 'your query'")
