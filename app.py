import streamlit as st
import os
import tempfile
from data_ingestion import process_pdf_and_store
from retriever import retrieve_context
from defense_layers import (
    check_risk,
    check_relevance,
    generate_answer,
    audit_hallucination,
    check_compliance_and_tone
)

st.set_page_config(page_title="Secure Financial RAG", page_icon="🛡️", layout="wide")

st.title("🛡️ Secure Financial RAG System")
st.markdown("一個擁有 5 道資安防線的企業級金融問答系統")

# --- Sidebar: Data Ingestion ---
with st.sidebar:
    st.header("📂 資料庫管理")
    uploaded_files = st.file_uploader("上傳金融報告 (PDF)", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        if st.button("處理並寫入知識庫"):
            with st.spinner("處理中..."):
                for uploaded_file in uploaded_files:
                    # Save uploaded file to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        process_pdf_and_store(tmp_file_path)
                        st.success(f"✅ {uploaded_file.name} 已成功寫入！")
                    except Exception as e:
                        st.error(f"寫入 {uploaded_file.name} 失敗: {e}")
                    finally:
                        os.unlink(tmp_file_path)

# --- Main Chat Interface ---
st.header("💬 智能金融問答")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("請問關於金融報告的問題..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        # UI Elements for 5 Layers
        status_col, content_col = st.columns([1, 2])
        
        with status_col:
            st.subheader("🛡️ 防線狀態")
            layer1_ph = st.empty()
            layer2_ph = st.empty()
            layer3_ph = st.empty()
            layer4_ph = st.empty()
            layer5_ph = st.empty()
            
            layer1_ph.info("Layer 1: 意圖攔截器 (Risk Router) - 檢查中...")
            layer2_ph.write("Layer 2: 相關性過濾 (Relevance Checker) - 待命")
            layer3_ph.write("Layer 3: 限制生成 (Ground-Truth Generator) - 待命")
            layer4_ph.write("Layer 4: 幻覺稽核站 (Auditor) - 待命")
            layer5_ph.write("Layer 5: 合規語氣檢查 (Compliance & Tone) - 待命")

        with content_col:
            st.subheader("📝 處理細節")
            details_ph = st.empty()

        # Step 1: Risk Router
        risk_result = check_risk(prompt)
        if risk_result.get("status") == "block":
            layer1_ph.error(f"❌ Layer 1 攔截: {risk_result.get('reason')}")
            final_answer = f"**系統拒絕回答**：{risk_result.get('reason')}"
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            st.stop()
        layer1_ph.success("✅ Layer 1: 意圖安全")

        # Step 1.5: Retrieval
        layer2_ph.info("Layer 2: 相關性過濾 (Relevance Checker) - 檢索並檢查中...")
        context = retrieve_context(prompt)
        details_ph.text_area("檢索到的文本 (Context)", value=context, height=150)
        
        # Step 2: Relevance Checker
        relevance_result = check_relevance(prompt, context)
        if relevance_result.get("status") == "block":
            layer2_ph.error(f"❌ Layer 2 攔截: {relevance_result.get('reason')}")
            final_answer = f"**資料不足**：{relevance_result.get('reason')}"
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            st.stop()
        layer2_ph.success("✅ Layer 2: 發現相關資料")

        # Step 3: Ground-Truth Generator
        layer3_ph.info("Layer 3: 限制生成 (Ground-Truth Generator) - 生成草稿中...")
        draft = generate_answer(prompt, context)
        details_ph.text_area("生成草稿 (Draft)", value=draft, height=150)
        layer3_ph.success("✅ Layer 3: 草稿生成完畢")

        # Step 4: Auditor
        layer4_ph.info("Layer 4: 幻覺稽核站 (Auditor) - 比對數據中...")
        audit_result = audit_hallucination(draft, context)
        if not audit_result.get("pass", False):
            issue = audit_result.get('issue', '未知幻覺')
            layer4_ph.error(f"❌ Layer 4 攔截: 發現幻覺漏洞 ({issue})")
            final_answer = f"**系統拒絕回答**：發現可能的幻覺與捏造數據 ({issue})"
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            st.stop()
        layer4_ph.success("✅ Layer 4: 無發現幻覺數據")

        # Step 5: Compliance and Tone Checker
        layer5_ph.info("Layer 5: 合規語氣檢查 (Compliance & Tone) - 審查中...")
        compliance_result = check_compliance_and_tone(draft)
        if compliance_result.get("status") == "block":
            layer5_ph.error(f"❌ Layer 5 攔截: {compliance_result.get('reason')}")
            final_answer = f"**系統拒絕回答**：觸發合規紅線"
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            st.stop()
        
        layer5_ph.success("✅ Layer 5: 合規並加上免責聲明")
        
        final_answer = compliance_result.get("revised_text", draft)
        details_ph.empty() # Clear details
        st.markdown(f"### 最終回覆\n{final_answer}")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
