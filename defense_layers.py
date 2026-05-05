import os
import json
from dotenv import load_dotenv
from groq import Groq

# Initialize Groq Client
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL_NAME = "llama-3.1-8b-instant"

def check_risk(query: str) -> dict:
    """
    Defense Layer 1: Risk Router
    Blocks risky intents such as investment advice, stock price predictions, etc.
    """
    system_prompt = """你是一個資安意圖攔截器。
嚴格阻擋任何與「投資建議」、「股價預測」、「是否買進」、「是否賣出」、「All-in」、「買入」、「賣出」、「股票推薦」相關的意圖。
如果使用者的問題包含任何上述風險意圖或詢問是否該投資，請回傳 JSON 格式：{"status": "block", "reason": "拒絕提供投資建議"}
如果安全且只是客觀詢問數據或事實，請回傳 JSON 格式：{"status": "pass"}
請務必只回傳合法的 JSON 字串，不要包含任何其他解釋或文字。"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        return {"status": "block", "reason": "系統風險檢查異常"}


def check_relevance(query: str, context: str) -> dict:
    """
    Defense Layer 2: Relevance Checker
    Ensures the retrieved context contains relevant information to answer the query.
    """
    if not context.strip():
        return {"status": "block", "reason": "未檢索到任何參考資料"}

    system_prompt = f"""你是一個寬容的相關性檢查員。
你的任務是判斷提供的 Context 是否包含與使用者問題「相關」的資訊。
請注意：
1. 台灣民國紀年 114 年即為西元 2025 年，113 年為 2024 年。
2. 只要 Context 內容（如公司名稱、財報數據、營運狀況等）與問題有部分關聯，就應該讓它通過。
3. 如果資訊不夠完整，後續的生成器會處理，你不需在此阻擋。
只有當 Context 「完全」與問題無關（例如問 A 公司卻只有 B 公司的資料），才回傳 JSON 格式：{{"status": "block", "reason": "資料庫中沒有相關資訊"}}
否則請回傳 JSON 格式：{{"status": "pass"}}
請務必只回傳合法的 JSON 字串。

Context:
{context}"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Exception in check_relevance: {e}")
        # Default to pass to avoid over-blocking, or block for extreme safety.
        # Here we choose block for strict financial compliance.
        return {"status": "block", "reason": "相關性檢查系統異常"}


def generate_answer(query: str, context: str) -> str:
    """
    Defense Layer 3: Ground-Truth Generator
    Generates answer strictly based on the context.
    """
    system_prompt = f"""你是一個精確的金融問答系統。
強制要求：
1. 只能根據提供的 Context 回答。若使用者詢問西元年份 (如 2025 年)，請注意 Context 中的台灣民國紀年 (如 114 年) 即為對應年份 (2025=114, 2024=113)。
2. 每句話結尾盡量附上來源，格式為 [來源_段落]。
3. 若 Context 沒提到，必須回答「資料不足」，不可捏造資訊。

Context:
{context}"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content


def audit_hallucination(generated_text: str, context: str) -> dict:
    """
    Defense Layer 4: Hallucination Auditor
    Checks if any numbers or proprietary names in the generated text are made up.
    """
    system_prompt = f"""你是一個精準的數據稽核員。
你的任務是比對 generated_text 裡面的所有「數字」、「數據」與「專有名詞」，是否確實存在於 Context 中。
請遵守以下寬容規則：
1. 忽略文本中的參考來源標籤 (例如 [XXX_Page_0] 或 [來源_段落])。
2. 允許數字單位的格式差異 (例如 $15.36 與 15.36元，或者表格中的 15.36 都是相等的)。
3. 只要 Context 中能合理推導出該數值與該指標的對應關係，即算通過，不需要字字句句完全一致。

只有當你「非常確定」這個數字是憑空捏造、或是被張冠李戴時，才回傳 JSON 格式：{{"pass": false, "issue": "抓到的錯誤細節"}}
如果數據都吻合或有合理來源，回傳 JSON 格式：{{"pass": true}}
請務必只回傳合法的 JSON 字串。

Context:
{context}"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generated Text:\n{generated_text}"}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"pass": False, "issue": "稽核系統解析異常"}


def check_compliance_and_tone(generated_text: str) -> dict:
    """
    Defense Layer 5: Compliance & Tone Checker
    Ensures the final output tone is objective, not providing financial advice, and contains a disclaimer.
    """
    system_prompt = """你是一個金融合規審查員。
請確保以下回答沒有煽動性情緒、沒有給予明確的投資建議（如買進、賣出、目標價）。
如果你認為文本合規，請在原文本後加上免責聲明「\n\n**免責聲明**：本資訊僅供參考，不構成任何投資建議。」然後回傳。
請回傳 JSON 格式：
{"status": "pass", "revised_text": "加上免責聲明後的完整文本"}
若文本極度不合規且無法修正，回傳 JSON 格式：
{"status": "block", "reason": "觸發合規紅線"}
請只回傳合法 JSON 字串。"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": generated_text}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"status": "block", "reason": "合規檢查系統異常"}
