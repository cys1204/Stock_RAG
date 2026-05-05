# 🛡️ Secure Financial RAG System

一個具備 **5 道資安防線**的企業級金融問答系統。本專案透過檢索增強生成 (RAG) 架構結合多層防護機制，確保大型語言模型 (LLM) 在回答金融財報問題時，能嚴格遵守合規性、避免幻覺（Hallucination），並拒絕提供任何投資建議。
<img width="1917" height="924" alt="image" src="https://github.com/user-attachments/assets/b4d88834-78ed-46f4-943c-72755be5cda0" />

## 🌟 系統特色 (Current Features)

本系統內建了 5 層防禦機制，每一層都獨立由 LLM 進行稽核或生成：

1. **Layer 1: 意圖攔截器 (Risk Router)**
   - 嚴格阻擋任何與「投資建議」、「股價預測」、「買賣決策」相關的意圖。
2. **Layer 2: 相關性過濾 (Relevance Checker)**
   - 確保從資料庫檢索到的文本，確實包含與使用者問題相關的資訊，避免硬答。
3. **Layer 3: 限制生成 (Ground-Truth Generator)**
   - 強制 LLM **只能**依據提供的上下文 (Context) 進行回答，若無資訊則必須回答「資料不足」。
4. **Layer 4: 幻覺稽核站 (Auditor)**
   - 交叉比對生成文本與原始文本，嚴格抓取是否出現捏造的「數字」、「數據」與「專有名詞」。
5. **Layer 5: 合規語氣檢查 (Compliance & Tone)**
   - 確保最終回答客觀、無煽動性，並強制於文末加上「免責聲明」。

### 📚 資料處理與檢索策略 (Data Ingestion & Retrieval Strategy)
為了確保財報資料能被精確檢索，系統在背後進行了嚴謹的資料處理與優化：
1. **多格式文檔解析**: 使用 LangChain 的 `PyPDFLoader` 與 `TextLoader` 支援 PDF 與 TXT 檔案上傳。
2. **Chunking 策略 (文本切分)**: 
   - 採用 `RecursiveCharacterTextSplitter`，主要分隔符號 (Separators) 依序設定為 `["\n\n", "\n", "。", "，", " ", ""]`。
   - **Chunk Size**: 1000 字元；**Chunk Overlap**: 200 字元。
   - 這樣的設定能確保長篇幅的財報段落被切分時，優先從段落或句號斷開，並透過重疊保留上下文，避免語意流失。
3. **Embedding 模型與向量資料庫**: 
   - **Embedding 模型**: 採用 `paraphrase-multilingual-MiniLM-L12-v2`。這是一個輕量且對多語言（包含繁體中文）支援度極佳的模型，能準確捕捉金融術語與中文的語意特徵。
   - **向量資料庫**: 採用 `ChromaDB` 作為 Persistent Storage，並使用餘弦相似度 (`cosine similarity`) 進行本地端的高效能檢索。
4. **Retriever 的優化與擴充**: 
   - 目前採用標準的 **Top-K 相似度檢索 (K=4)**。
   - **優化解法**：未來可導入 **HyDE (Hypothetical Document Embeddings)**，先讓 LLM 生成一段「假想答案」再進行向量比對，能大幅提升檢索命中率。此外，針對財報特徵，強制加入 **Metadata Filtering** (如限制檢索特定年份、季度) 可有效解決「民國114年與2025年」的時間點混淆問題。

### 📊 系統評估與測試 (Evaluation & Benchmarks)
要驗證 RAG 系統與五道防線的表現，本專案的評估機制如下：
1. **內建端到端測試 (`test_pipeline.py`)**:
   - 專案內建自動化測試腳本，針對「正常提問」、「惡意投資誘導」、「無關提問」進行封閉測試，確保每一道防線 (如 Risk Router 與 Auditor) 都能精準作動。
2. **未來建議導入的評估框架 (Benchmarks)**:
   - **RAGAS (RAG Assessment)**: 強烈建議導入此工具來量化評估 Retriever 的精準度 (Context Precision / Recall) 以及 Generator 的品質 (Answer Relevancy / Faithfulness)。
   - **TruLens**: 可用於追蹤防線攔截的軌跡，並利用 LLM-as-a-Judge 的方式，自動為系統的幻覺率 (Hallucination) 與合規性 (Compliance) 進行大規模批量打分。

### 核心技術棧
* **UI 介面**: Streamlit (`app.py`)
* **LLM 引擎**: Groq API (`llama-3.1-8b-instant`)
* **向量資料庫**: ChromaDB (Persistent Storage)
* **文本嵌入**: SentenceTransformers (`paraphrase-multilingual-MiniLM-L12-v2`，針對中文優化)
* **文檔解析**: LangChain (`PyPDFLoader`, `TextLoader`)

---

## 🚀 快速開始 (Quick Start)

### 1. 安裝依賴套件
```bash
pip install -r requirements.txt
```

### 2. 環境變數設定
請複製 `.env.example` 並重新命名為 `.env`，接著填入您的 Groq API Key：
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. 啟動服務
```bash
streamlit run app.py
```
啟動後，可於左側邊欄上傳 PDF 或 TXT 格式的金融報告，並在右側進行安全問答。

---

## 🛠️ 未來優化方向 (Areas for Improvement)

雖然系統架構已完善，但在面對真實世界複雜的財報數據時，仍有以下地方需要持續修正與升級：

1. **📊 表格數據解析 (Table Extraction)**
   - **現狀**：目前使用 `RecursiveCharacterTextSplitter` 依照字元長度切分文本。
   - **問題**：對於財報中常見的「綜合損益表」、「資產負債表」等複雜表格，容易因為強制切斷而喪失行列對應關係。
   - **解法**：建議未來引入 `pdfplumber` 或 `Unstructured` 等專門針對表格的解析套件，將表格獨立提取並轉化為 Markdown 格式存放。

2. **🔍 進階檢索策略 (Advanced Retrieval)**
   - **現狀**：目前使用基礎的 Top-K 向量相似度搜尋。
   - **問題**：使用者若提問「2025年第一季EPS」，可能找不出明確數據，因為財報內可能寫作「114年第1季」。
   - **解法**：可加入 Metadata 過濾 (例如按年份、季度篩選) 或引入 HyDE (Hypothetical Document Embeddings) 技術來提升精準度。

3. **⚙️ 錯誤處理與 API 限制 (Error Handling & Rate Limits)**
   - **現狀**：為了確保資安，如果 LLM 解析 JSON 失敗，系統預設為「阻擋 (Block)」。使用者輸入一次問題，會觸發至少 4 次 Groq API 請求。
   - **問題**：雖然安全，但若 Groq API 不穩或達到 Rate Limit，容易造成誤擋 (False Positives)。
   - **解法**：實作 Retry 機制 (重試邏輯) 以及適當的錯誤降級提示。

4. **⚡ 回應速度優化 (Latency Optimization)**
   - **現狀**：五道防線採循序執行。
   - **解法**：Layer 1 與文檔檢索可以設計成「非同步 (Async) 併發執行」，進一步縮短使用者的等待時間。

---

## 📂 專案結構
```
Stock_RAG/
├── app.py                # Streamlit 主程式
├── data_ingestion.py     # 處理 PDF/TXT 檔案並寫入 ChromaDB
├── retriever.py          # 負責從 ChromaDB 中檢索相似文檔
├── defense_layers.py     # 定義 5 道資安防禦層邏輯
├── test_pipeline.py      # 終端機純文字測試腳本
├── chroma_db/            # ChromaDB 本地向量資料庫存放位置
├── reports/              # 測試用財報 PDF 資料夾
├── requirements.txt      # 專案依賴套件
└── .env                  # 環境變數設定檔
```
