import os
import json
from dotenv import load_dotenv
from groq import Groq

# 1. 環境與配置 (Setup & Mock Data)
load_dotenv()

# 初始化 Groq Client，請確保 .env 檔案中有 GROQ_API_KEY
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def mock_retrieve(query):
    """
    模擬向量資料庫檢索 (Mock Retriever)
    """
    return "台積電 2026 Q1 法說會重點：毛利率 53%、營收成長 15%、資本支出 300 億美元。 [2026_Q1_法說會_段落2]"

# 2. 第 1 道防線：意圖攔截器 (Risk Router)
def check_risk(query):
    system_prompt = """你是一個資安意圖攔截器。
嚴格阻擋「投資建議」、「股價預測」、「是否買進」、「是否賣出」、「All-in」等意圖。
如果使用者的問題包含上述風險意圖，請回傳 JSON 格式：{"status": "block", "reason": "拒絕提供投資建議"}
如果安全，請回傳 JSON 格式：{"status": "pass"}
請務必只回傳合法的 JSON 字串，不要包含任何其他解釋或文字。"""
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        # 如果解析失敗，為了安全起見預設阻擋
        return {"status": "block", "reason": "系統風險檢查異常"}

# 3. 第 2 道防線：生成與強制引用 (Generator & Grounding)
def generate_answer(query, context):
    system_prompt = f"""你是一個精確的金融問答系統。
強制要求：
1. 只能根據提供的 context 回答。
2. 每句話結尾必須附上來源，格式為 [來源_段落]。
3. 若 context 沒提到，必須回答「資料不足」，不可捏造資訊。

Context:
{context}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content

# 4. 第 3 道防線：幻覺稽核站 (Auditor)
def audit_hallucination(generated_text, context):
    system_prompt = f"""你是一個無情的數據稽核員。
你的任務是比對 generated_text 裡面的所有「數字」、「數據」與「專有名詞」，是否 100% 存在於 context 中。
如果有任何捏造的數字（例如 context 沒有提到的 EPS、不同季度的數據等），回傳 JSON 格式：{{"pass": false, "issue": "抓到的錯誤細節"}}
若完全吻合，回傳 JSON 格式：{{"pass": true}}
請務必只回傳合法的 JSON 字串，不要包含任何其他文字。

Context:
{context}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generated Text:\n{generated_text}"}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        # 如果解析失敗，為了安全起見預設為幻覺
        return {"pass": False, "issue": "稽核系統解析異常"}

# 5. 執行主程式 (Main Workflow Pipeline)
def main(user_input):
    print(f"\n{'='*60}")
    print(f"使用者提問: {user_input}")
    print(f"{'-'*60}")
    
    # Step 1: 檢查風險
    print("[1. Risk Router] 檢查風險中...")
    risk_result = check_risk(user_input)
    if risk_result.get("status") == "block":
        print(f"⚠️ 攔截：{risk_result.get('reason')}")
        return
    print("✅ 通過風險檢查")
    
    # Step 2: 撈取資料
    print("\n[2. Mock Retriever] 撈取資料...")
    context = mock_retrieve(user_input)
    print(f"📄 撈取到的文本: {context}")
    
    # Step 3: 生成草稿
    print("\n[3. Generator] 生成草稿中 (不直接輸出給使用者)...")
    draft = generate_answer(user_input, context)
    print(f"📝 內部草稿內容: \n{draft}")
    
    # Step 4: 數字稽核
    print("\n[4. Auditor] 進行幻覺與數字稽核中...")
    audit_result = audit_hallucination(draft, context)
    if not audit_result.get("pass", False):
        print(f"\n⚠️ 觸發資安防禦：發現幻覺漏洞 ({audit_result.get('issue', '未知錯誤')})")
        print("❌ 拒絕給出答案")
        return
         
    print(f"\n✅ 最終安全回答:\n{draft}")

if __name__ == "__main__":
    # 測試案例
    test_cases = [
        "請問台積電 Q1 資本支出是多少？",
        "台積電現在適合 All-in 嗎？",
        "請問台積電 Q1 營收成長率與 Q2 的預估 EPS 是多少？"
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n\n\n>>> 執行測試案例 {i} <<<")
        main(case)
