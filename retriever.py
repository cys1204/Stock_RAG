import os
import json
from dotenv import load_dotenv
from groq import Groq
from data_ingestion import get_chroma_collection

load_dotenv()

def parse_query_intent(query: str) -> dict:
    """Uses LLM to extract company name from the query for Metadata filtering."""
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        system_prompt = (
            "你是一個意圖分析助手。請從以下使用者問題中擷取「公司名稱」或「公司簡稱」。\n"
            "如果有找到，請輸出 JSON 格式：{\"company\": \"公司名稱\"}。\n"
            "如果沒有提到任何公司，請輸出：{\"company\": \"\"}。\n"
            "請只輸出合法的 JSON，不要有任何 Markdown 或其他文字。"
        )
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"問題：{query}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        result = response.choices[0].message.content
        data = json.loads(result)
        return {"company": data.get("company", "")}
    except Exception as e:
        print(f"Query intent parsing failed: {e}")
        return {"company": ""}

def generate_hyde_document(query: str) -> str:
    """Uses LLM to generate a hypothetical document based on the query."""
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        system_prompt = (
            "你是一個金融分析師。請撰寫一段假想的財務報告段落來回答以下問題。\n"
            "這段文字將用於向量檢索，請盡量使用專業的財報術語與正式的行文風格。\n"
            "不要寫引言，直接給出假想的財報內容即可。"
        )
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"問題：{query}"}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"HyDE generation failed: {e}")
        return query # Fallback to original query if HyDE fails

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

    # 1. Parse intent for Metadata Filtering
    intent = parse_query_intent(query)
    where_filter = None
    if intent["company"]:
        where_filter = {"company": intent["company"]}
        print(f"[Retriever] Applied Metadata Filter: {where_filter}")

    # 2. Generate HyDE Document
    hyde_doc = generate_hyde_document(query)
    print(f"[Retriever] Generated HyDE Document (first 50 chars): {hyde_doc[:50]}...")

    # 3. Query ChromaDB
    results = None
    if where_filter:
        try:
            results = collection.query(query_texts=[hyde_doc], n_results=top_k, where=where_filter)
        except Exception as e:
            print(f"[Retriever] Filtered query error: {e}")
            
    # ONLY fallback to unfiltered search if we DID NOT apply a filter
    if not where_filter:
        print("[Retriever] No company filter applied. Searching globally...")
        results = collection.query(query_texts=[hyde_doc], n_results=top_k)

    if not results['documents'] or not results['documents'][0]:
        return ""

    context_parts = []
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    for i, doc in enumerate(documents):
        source = metadatas[i].get("source", "Unknown")
        company = metadatas[i].get("company", "Unknown")
        # Ensure we only have the filename
        source_name = os.path.basename(source)
        page = metadatas[i].get("page", 0)
        
        context_parts.append(f"[{source_name}_Page_{page}_Company_{company}]\n{doc}")

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
