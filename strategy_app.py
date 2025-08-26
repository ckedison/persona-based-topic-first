# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io
import re

# --- 頁面設定 ---
st.set_page_config(
    page_title="互動式策略儀表板 (優化版)",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 初始化 Session State ---
# 確保所有需要的 session state 鍵都已初始化
def initialize_session_state():
    defaults = {
        'persona_df': None,
        'query_fan_out_df': None,
        'matched_personas': None,
        'api_key_configured': False,
        'strategy_text': None,
        'funnel_text': None,
        'topic': "",
        'persona_prompt': "",
        'pasted_persona_csv': "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- 狀態重設回呼函式 ---
def reset_analysis_state():
    """當核心輸入（主題、Persona）改變時，重設後續的分析結果"""
    st.session_state.matched_personas = None
    st.session_state.strategy_text = None
    st.session_state.funnel_text = None
    st.toast("偵測到輸入變更，已重設分析狀態。")

def reset_query_fan_out_state():
    """當 Query Fan Out 檔案改變時，重設相關狀態"""
    st.session_state.query_fan_out_df = None
    reset_analysis_state()

# --- Prompt Engineering 函式 ---
# (這部分的功能與原版相同，僅集中放置)

def create_iterative_persona_prompt(topic):
    """(新增) 為 AI 生成 Persona 建立 Prompt"""
    return f"""
請扮演一位專業的市場研究員與內容策略師。
我的核心業務主題是：「{topic}」。

你的任務是為這個主題構思 3-5 個不同且具體的目標受眾人物誌 (Persona)。這些 Persona 應該涵蓋從新手到專家的不同需求層級。

請嚴格遵循以下 CSV 格式輸出，包含標頭，並且不要有任何其他的開頭或結尾文字。每一筆資料的欄位內容請用雙引號 `"` 包覆。

```csv
"persona_name","summary","goals","pain_points","keywords","preferred_formats"
"範例人物誌1 (例如：焦慮的新手媽媽)","簡短描述這個 Persona 的背景和特徵。","他們想達成的 1-2 個主要目標。","他們在達成目標時遇到的 1-2 個主要困難或痛點。","他們可能會用來搜尋相關資訊的 3-5 個關鍵字，用逗號分隔。","他們偏好的內容格式，例如：短影音, 圖文卡, 部落格文章, Podcast，用逗號分隔。"
"範例人物誌2 (例如：尋求效率的職場人士)","...","...","...","...","..."
... (直到 3-5 筆)
```

**生成指南:**
- **persona_name:** 給予一個生動且能反映其身份的名稱。
- **summary:** 簡潔地描述其背景、動機與現況。
- **goals:** 具體說明他們圍繞「{topic}」這個主題希望達成的目標。
- **pain_points:** 他們在實踐過程中遇到的主要阻礙或困惑。
- **keywords:** 他們在 Google 搜尋時會使用的字詞。
- **preferred_formats:** 他們最喜歡消費的內容類型。

請確保生成的 Persona 具有代表性且彼此之間有明顯區隔。請開始生成。
"""

def create_query_fan_out_prompt(topic):
    """為 AI 生成 Query Fan Out 建立 Prompt"""
    return f"""
請扮演一位資深的 SEO 與內容策略專家。我的核心主題是：「{topic}」。
你的任務是為這個主題進行「Query Fan Out」分析，生成 15 個用戶可能會搜尋的相關查詢 (Query)。
請嚴格遵循以下 CSV 格式輸出，包含標頭，並且不要有任何其他的開頭或結尾文字。每一筆資料的欄位內容請用雙引號 `"` 包覆，以避免格式錯誤。
```csv
"query","type","user_intent","reasoning"
"範例查詢1","範例類型1","範例意圖1","範例理由1"
"範例查詢2","範例類型2","範例意圖2","範例理由2"
... (直到第15筆)
```
**生成指南:**
- **query:** 具體的用戶搜尋字詞。
- **type:** 查詢的類型，請從以下選項中選擇：[問題 (Question), 比較 (Comparison), 資訊 (Informational), 商業 (Commercial), 導航 (Navigational)]。
- **user_intent:** 總結用戶進行此搜尋背後的真實意圖。
- **reasoning:** 簡要說明為什麼這個查詢與核心主題「{topic}」相關。
請確保生成的查詢涵蓋不同的類型與用戶意圖，以展現主題的全貌。請開始生成。
"""

def create_dynamic_prompt(topic, selected_personas_df, query_fan_out_df=None):
    """根據主題和選擇的 Persona 動態生成 Prompt (優化版)"""
    persona_details = ""
    for index, row in selected_personas_df.iterrows():
        persona_details += f"""
### 人物誌 (Persona): {row['persona_name']}
- **核心摘要:** {row.get('summary', '無')}
- **主要目標:** {row.get('goals', '無')}
- **主要痛點:** {row.get('pain_points', '無')}
- **偏好內容格式:** {row.get('preferred_formats', '無')}
"""

    query_fan_out_section = ""
    idea_format_instruction = ""
    idea_structure = ""

    if query_fan_out_df is not None and not query_fan_out_df.empty:
        query_fan_out_section = f"""
另外，請務必參考以下由 SEO 專家分析的「Query Fan Out」資料，這代表了用戶在搜尋此主題時的真實意圖與變化：
```
{query_fan_out_df.to_markdown(index=False)}
```
"""
        idea_format_instruction = """(請提供 3-5 個**緊扣上述「連結分析」**的具體內容點子。**每一個點子都必須明確對應到 Query Fan Out 資料中一個具體的 'query' 或 'user_intent'**。每一個點子都必須包含「主題/標題方向」、「對應的用戶查詢」、「建議格式」和「理由」。)"""
        idea_structure = """
* **點子一：**
    * **主題/標題方向:** [一個能直接反映「連結分析」的具體標題]
    * **對應的用戶查詢:** [從 Query Fan Out 中選擇一個最相關的 query/intent]
    * **建議格式:** [從 Persona 偏好格式中挑選]
    * **理由:** [說明為什麼這個點子和格式能有效**回應對應的用戶查詢**並解決 Persona 的問題]
"""
    else:
        idea_format_instruction = """(請提供 3-5 個**緊扣上述「連結分析」**的具體內容點子。每一個點子都必須包含「主題/標題方向」、「建議格式」和「理由」。)"""
        idea_structure = """
* **點子一：**
    * **主題/標題方向:** [一個能直接反映「連結分析」的具體標題]
    * **建議格式:** [從 Persona 偏好格式中挑選]
    * **理由:** [說明為什麼這個點子和格式能有效解決 Persona 在此主題下的特定問題]
"""

    return f"""
請扮演一位頂尖的內容策略顧問，擁有敏銳的用戶洞察力。我的核心主題是：「{topic}」。
你的任務是為以下的人物誌 (Persona) 規劃一份**高度相關且具體**的內容策略。
{query_fan_out_section}
這是我要你分析的人物誌資料：
{persona_details}
請為 **每一個** 人物誌提供一份獨立的策略建議。在規劃時，你必須深度思考「核心主題」、「Query Fan Out (如果提供)」與「Persona 的痛點/目標」之間的**交集**，並以此交集作為所有內容點子的出發點。
請嚴格遵循以下格式輸出，使用 Markdown 語法：
---
### **針對「[人物誌姓名]」的內容策略**
**1. 主題與 Persona 連結分析 (Topic-Persona Nexus):**
(請在此用 2-3 句話，精準分析「{topic}」這個主題，如何能有效解決此 Persona 的核心痛點或幫助他達成目標。**這是最重要的部分，請務必具體說明連結點。**)
**2. 核心溝通角度 (Core Angle):**
(基於以上的連結分析，總結出一個最能打動此 Persona 的核心溝通切角。)
**3. 內容點子與格式建議 (Content Ideas & Formats):**
{idea_format_instruction}
{idea_structure}
---
### **總結：內容產製清單 (Content Production Checklist)**
現在，請扮演一位**內容製作總監**。請回顧以上**所有**為不同 Persona 生成的內容點子，並將它們整合成一個清晰的總表。
這個表格的目的是讓團隊一目了然地知道總共需要製作哪些類型的內容，以及每個類型有哪些具體的點子。
請遵循以下表格格式，將**相似的「建議格式」**的點子歸類在一起：
| 內容格式 (Media Format) | 主題/標題方向 (Topic/Title Ideas) |
| :--- | :--- |
| **[例如：YouTube 深度影片]** | - [標題方向A]<br>- [標題方向B]<br>- [標題方向C] |
| **[例如：Podcast]** | - [標題方向D]<br>- [標題方向E] |
| **[例如：IG 圖文卡]** | - [標題方向F] |
請確保表格完整涵蓋了前面提到的所有點子。
"""

def create_funnel_prompt(topic, strategy_text, conversion_goal, query_fan_out_df=None):
    """根據初步策略和轉換目標生成行銷漏斗策略的 Prompt"""
    query_fan_out_section = ""
    if query_fan_out_df is not None and not query_fan_out_df.empty:
        query_fan_out_section = f"""
在規劃時，請優先考慮以下「Query Fan Out」資料中，具有高商業意圖或能解決深度問題的查詢，將其融入你的漏斗策略中：
```
{query_fan_out_df.to_markdown(index=False)}
```
"""

    conversion_goal_section = f"""
**重要：最終轉換目標**
請將以下的具體產品/服務資訊作為你設計「轉換階段 (BOFU)」內容與 CTA 的最終目標：
- **產品/服務名稱:** {conversion_goal.get('name', '未提供')}
- **期望用戶完成的動作:** {conversion_goal.get('action', '未提供')}
- **最終導向的目標網址:** {conversion_goal.get('url', '未提供')}
- **產品/服務簡介:** {conversion_goal.get('desc', '未提供')}
請確保漏斗的最後一步能有效地將用戶引導至此目標。
"""

    return f"""
請扮演一位頂尖的數位行銷策略總監 (Head of Digital Strategy)，專精於設計高轉換率的內容行銷漏斗。
我的核心主題是：「{topic}」。
{query_fan_out_section}
{conversion_goal_section}
這是一份由 AI 內容策略顧問針對不同 Persona 生成的初步內容點子清單：
```markdown
{strategy_text}
```
你的任務是，將這些零散的點子，整合成一個**環環相扣、無縫引導**的完整行銷活動。
請嚴格遵循以下步驟與格式輸出：
---
### **整合行銷漏斗策略："{topic}"**
**📈 總體策略與用戶旅程 (Overall Strategy & User Journey):**
(請在此以故事線的方式，清晰描述一個典型用戶從接觸第一個內容(認知)，到最後完成購買(轉換)的完整路徑。明確指出每一個階段的轉換目標和引導機制。)
---
### **1. 認知階段 (Awareness - Top of Funnel)**
*目標：透過高價值、易擴散的內容，大規模吸引對此主題感興趣的潛在用戶，建立品牌專業形象。*
**➡️ 內容點子 1 (主打):** [從清單中選擇最適合引流的內容點子]
   - **目標 Persona:** [此點子主要針對的 Persona]
   - **引流與擴散策略:** [例如：針對此主題投放 Instagram/Facebook 廣告；優化 SEO 關鍵字「...」；與親子KOL合作推廣此內容]
   - **➡️ 轉換至下一階段的 CTA (Call-to-Action):** **(此為重點)** [設計一個明確的行動呼籲，將用戶從這個認知內容，引導至考慮階段的內容。例如：「想知道如何實際應用嗎？點擊連結，免費下載我們的『XXX實踐手冊』！」]
---
### **2. 考慮階段 (Consideration - Middle of Funnel)**
*目標：透過更深入、更具體的內容，解決用戶的核心痛點，建立信任感，並獲取潛在客戶名單 (Leads)。*
**➡️ 內容點子 2 (主打):** [從清單中選擇最適合建立信任/獲取名單的內容點子，例如電子書、網路研討會、深度指南]
   - **目標 Persona:** [此點子主要針對的 Persona]
   - **接收流量來源:** [明確說明此內容的流量主要來自哪個認知階段的內容]
   - **價值交換設計 (Lead Magnet):** [例如：設計成一份精美的 PDF 電子書，用戶需提供 Email 才能下載。]
   - **➡️ 轉換至下一階段的 CTA (Call-to-Action):** **(此為重點)** [在用戶獲取此內容後，設計後續的引導路徑。例如：「下載手冊後，我們將在三天後寄送一封郵件，與您分享如何將手冊內容應用在...，並提供一個專屬的訂閱優惠。」]
---
### **3. 轉換階段 (Conversion - Bottom of Funnel)**
*目標：臨門一腳，透過直接的價值主張與誘因，促使用戶完成最終購買決策。*
**➡️ 內容點子 3 (主打):** [從清單中選擇最適合導購的內容點子，例如產品比較、用戶見證、優惠活動頁]
   - **目標 Persona:** [此點子主要針對的 Persona]
   - **接收流量來源:** [明確說明此內容的流量主要來自哪個考慮階段的內容或後續的 Email/LINE 行銷]
   - **導購與行動呼籲 (CTA) 設計:** [設計強而有力的 CTA，**務必結合前面提供的產品資訊與目標網址**。例如：「立即訂閱『{conversion_goal.get('name', '我們的服務')}』，解鎖所有專家內容！點擊前往：{conversion_goal.get('url', '#')}」]
---
**📊 總結：用戶旅程地圖**
(請用流程圖的方式，總結從 TOFU 到 BOFU 的轉換路徑)
* **[認知內容]** (例如: IG Reels 短影音) → **CTA:** "留言+1索取完整指南"
* → **[考慮內容]** (例如: 私訊發送 PDF 指南) → **CTA:** "指南中附有專屬訂閱優惠連結"
* → **[轉換內容]** (例如: 優惠訂閱頁面) → **最終目標:** 完成訂閱
"""


# --- 核心功能函式 (帶有快取功能) ---

@st.cache_data
def get_gemini_response(_prompt, _api_key):
    """通用的 Gemini API 呼叫函式，帶有快取功能"""
    try:
        genai.configure(api_key=_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(_prompt)
        return response.text
    except Exception as e:
        # 在實際應用中，這裡可以記錄更詳細的錯誤日誌
        st.error(f"呼叫 API 時發生錯誤: {e}")
        return None

@st.cache_data
def generate_embeddings(_texts, _api_key, task_type="RETRIEVAL_DOCUMENT"):
    """生成 Embeddings 的快取函式"""
    try:
        genai.configure(api_key=_api_key)
        result = genai.embed_content(
            model='models/text-embedding-004',
            content=_texts,
            task_type=task_type
        )
        return result['embedding']
    except Exception as e:
        st.error(f"生成 Embeddings 時發生錯誤: {e}")
        return None

def parse_csv_from_ai(text_response):
    """從 AI 的回應中解析出 CSV 內容"""
    if not text_response:
        return None
    # 使用正則表達式尋找 ```csv ... ``` 區塊
    match = re.search(r'```csv\n(.*?)\n```', text_response, re.DOTALL)
    if match:
        csv_text = match.group(1)
    else:
        # 如果找不到區塊，就直接使用整個文字，並去除頭尾空白
        csv_text = text_response.strip()

    try:
        csv_io = io.StringIO(csv_text)
        df = pd.read_csv(csv_io)
        return df
    except Exception as e:
        st.error(f"解析 AI 生成的 CSV 內容失敗: {e}")
        st.info("AI 回應內容如下：")
        st.code(text_response)
        return None

# --- Streamlit 介面佈局 ---

st.title("🎯 互動式策略儀表板 (優化版)")
st.markdown("上傳您的 Persona，讓 AI 理解語意並為您打造主題優先的內容策略")

# --- 側邊欄設定面板 ---
with st.sidebar:
    st.header("⚙️ 設定面板")

    api_key = st.text_input("請輸入您的 Gemini API 金鑰", type="password", help="[點此取得您的 API 金鑰](https://aistudio.google.com/app/apikey)")

    if api_key:
        st.session_state.api_key_configured = True
        st.info("API 金鑰已設定。")
    else:
        st.session_state.api_key_configured = False


    st.markdown("---")
    st.subheader("1. 輸入核心主題")
    topic = st.text_input(
        "輸入您想規劃內容的核心主題",
        placeholder="例如：青少年理財教育",
        on_change=reset_analysis_state,
        key='topic'
    )

    st.markdown("---")
    st.subheader("2. Persona 資料")

    # 區塊 A: 上傳檔案
    uploaded_persona_file = st.file_uploader(
        "上傳 Persona CSV 檔案 (建議)",
        type="csv",
        key="persona_uploader",
        on_change=reset_analysis_state
    )
    if uploaded_persona_file:
        try:
            df = pd.read_csv(uploaded_persona_file)
            required_headers = ['persona_name', 'summary', 'goals', 'pain_points', 'keywords', 'preferred_formats']
            missing_headers = [h for h in required_headers if h not in df.columns]

            if missing_headers:
                st.error(f"Persona CSV 檔案缺少欄位: {', '.join(missing_headers)}")
                st.session_state.persona_df = None
            else:
                st.session_state.persona_df = df
                st.success(f"成功載入 {len(df)} 筆 Persona 資料！")
        except Exception as e:
            st.error(f"Persona 檔案讀取失敗：{e}")
            st.session_state.persona_df = None

    # 區塊 B: AI 輔助生成
    with st.expander("需要 AI 協助生成 Persona 嗎？"):
        if st.button("產生 Persona 生成指令", key="gen_persona_prompt"):
            if not st.session_state.topic:
                st.warning("請先輸入核心主題。")
            else:
                st.session_state.persona_prompt = create_iterative_persona_prompt(st.session_state.topic)

        if st.session_state.persona_prompt:
            st.text_area("1. 複製以下指令，並到您的 Gemini 介面執行", value=st.session_state.persona_prompt, height=200)
            pasted_persona_csv = st.text_area("2. 將 Gemini 生成的 CSV 結果貼於此處", height=150, key="pasted_persona_csv")

            if st.button("處理貼上的 Persona 資料", key="process_pasted_persona"):
                if pasted_persona_csv:
                    df = parse_csv_from_ai(pasted_persona_csv)
                    if df is not None:
                        st.session_state.persona_df = df
                        reset_analysis_state() # 重設狀態
                        st.success(f"成功處理 {len(df)} 筆貼上的 Persona 資料！")
                else:
                    st.warning("請先貼上資料。")

    st.markdown("---")
    st.subheader("3. Query Fan Out 資料 (選填)")
    uploaded_query_file = st.file_uploader(
        "上傳 Query Fan Out CSV 檔案",
        type="csv",
        key="query_uploader",
        on_change=reset_query_fan_out_state
    )

    if uploaded_query_file:
        try:
            df = pd.read_csv(uploaded_query_file)
            required_headers = ['query', 'type', 'user_intent', 'reasoning']
            if all(h in df.columns for h in required_headers):
                st.session_state.query_fan_out_df = df
                st.success(f"成功載入 {len(df)} 筆 Query Fan Out 資料！")
            else:
                st.error(f"Query Fan Out CSV 檔案缺少必要欄位。")
                st.session_state.query_fan_out_df = None
        except Exception as e:
            st.error(f"Query Fan Out 檔案讀取失敗：{e}")
            st.session_state.query_fan_out_df = None

    if st.session_state.query_fan_out_df is None:
        if st.button("📊 自動生成 Query Fan Out", use_container_width=True):
            if not st.session_state.api_key_configured or not st.session_state.topic:
                st.warning("請先輸入 API 金鑰和核心主題。")
            else:
                with st.spinner("正在為您自動生成相關查詢..."):
                    prompt = create_query_fan_out_prompt(st.session_state.topic)
                    response_text = get_gemini_response(prompt, api_key)
                    if response_text:
                        generated_qfo_df = parse_csv_from_ai(response_text)
                        if generated_qfo_df is not None:
                            st.session_state.query_fan_out_df = generated_qfo_df
                            st.success(f"已成功為您生成 {len(generated_qfo_df)} 筆相關查詢！")
                            st.rerun()

    st.markdown("---")
    if st.button("🔍 執行策略分析", use_container_width=True, type="primary"):
        if not st.session_state.api_key_configured:
            st.warning("請先輸入您的 API 金鑰。")
        elif not st.session_state.topic:
            st.warning("請輸入核心主題。")
        elif st.session_state.persona_df is None:
            st.warning("請先上傳或生成 Persona 資料。")
        else:
            with st.spinner("正在進行語意分析與匹配..."):
                # 1. 為 Persona 資料建立語意索引
                persona_df_processed = st.session_state.persona_df.copy()
                persona_df_processed['embedding_text'] = persona_df_processed['summary'].fillna('') + ' | ' + \
                                                         persona_df_processed['goals'].fillna('') + ' | ' + \
                                                         persona_df_processed['pain_points'].fillna('') + ' | ' + \
                                                         persona_df_processed['keywords'].fillna('')
                persona_texts = persona_df_processed['embedding_text'].tolist()
                persona_embeddings_list = generate_embeddings(persona_texts, api_key, "RETRIEVAL_DOCUMENT")

                if persona_embeddings_list:
                    # 2. 建立上下文 embedding
                    context_text = st.session_state.topic
                    if st.session_state.query_fan_out_df is not None:
                        queries = " ".join(st.session_state.query_fan_out_df['query'].fillna(''))
                        intents = " ".join(st.session_state.query_fan_out_df['user_intent'].fillna(''))
                        context_text += f" - 相關查詢與意圖: {queries} {intents}"

                    context_embedding_list = generate_embeddings([context_text], api_key, "RETRIEVAL_QUERY")

                    if context_embedding_list:
                        # 3. 計算餘弦相似度
                        context_embedding = np.array(context_embedding_list[0]).reshape(1, -1)
                        persona_embeddings = np.array(persona_embeddings_list)
                        similarities = cosine_similarity(context_embedding, persona_embeddings)[0]

                        persona_df_processed['score'] = similarities
                        matched = persona_df_processed.sort_values(by='score', ascending=False).head(10)
                        st.session_state.matched_personas = matched
                        st.session_state.strategy_text = None # 清空舊策略
                        st.session_state.funnel_text = None # 清空舊漏斗
                    else:
                        st.error("無法生成上下文 Embedding。")
                else:
                    st.error("無法生成 Persona Embeddings。")


# --- 主畫面顯示區 ---
if st.session_state.matched_personas is None:
    st.info("請在左側面板完成設定，點擊「執行策略分析」後，結果將顯示於此。")
else:
    st.markdown("---")
    st.subheader("4. 選擇相關 Persona")
    st.markdown("以下是根據您的主題與 Query Fan Out (若有) **語意關聯度**匹配出的 Persona。請勾選您想為其規劃策略的對象。")

    selected_indices = []
    if st.session_state.matched_personas.empty:
        st.warning("找不到符合條件的 Persona，請嘗試調整核心主題或檢查上傳的檔案。")
    else:
        for index, row in st.session_state.matched_personas.iterrows():
            cols = st.columns([0.1, 0.7, 0.2])
            is_selected = cols[0].checkbox("", key=f"persona_{index}", value=True) # 預設選取
            if is_selected:
                selected_indices.append(index)
            with cols[1]:
                st.markdown(f"**{row['persona_name']}**")
                st.caption(row['summary'])
            with cols[2]:
                st.metric(label="關聯度", value=f"{row['score']:.0%}")

    if selected_indices:
        st.markdown("---")
        if st.button("🚀 為選定對象生成初步策略", use_container_width=True):
            selected_df = st.session_state.matched_personas.loc[selected_indices]
            prompt = create_dynamic_prompt(st.session_state.topic, selected_df, st.session_state.query_fan_out_df)
            with st.spinner("🧠 AI 內容顧問正在生成初步點子..."):
                response_text = get_gemini_response(prompt, api_key)
                st.session_state.strategy_text = response_text
                st.session_state.funnel_text = None # 清空舊漏斗

    if st.session_state.strategy_text:
        st.markdown("---")
        st.subheader("5. AI 生成的初步內容策略")
        st.markdown(st.session_state.strategy_text)

        st.markdown("---")
        st.subheader("6. 整合行銷漏斗策略")

        with st.form(key='funnel_form'):
            st.markdown("**在生成最終漏斗前，請設定您的轉換目標：**")
            cols = st.columns(2)
            product_name = cols[0].text_input("產品/服務名稱", placeholder="例如：親子理財線上課")
            conversion_action = cols[1].selectbox("期望轉換動作", ['購買商品', '填寫表單', '預約諮詢', '訂閱服務', '下載App'])
            target_url = st.text_input("目標網址 (URL)", placeholder="https://example.com/product-page")
            product_desc = st.text_area("產品/服務簡介 (選填)", placeholder="簡要說明您的產品特色與價值")

            submit_button = st.form_submit_button(label="🧠 生成整合行銷漏斗策略", use_container_width=True, type="primary")

            if submit_button:
                if not product_name or not target_url:
                    st.warning("請填寫「產品/服務名稱」與「目標網址」。")
                else:
                    conversion_goal = {
                        "name": product_name,
                        "action": conversion_action,
                        "url": target_url,
                        "desc": product_desc
                    }
                    funnel_prompt = create_funnel_prompt(st.session_state.topic, st.session_state.strategy_text, conversion_goal, st.session_state.query_fan_out_df)
                    with st.spinner("👑 AI 行銷總監正在建構漏斗策略..."):
                        funnel_response = get_gemini_response(funnel_prompt, api_key)
                        st.session_state.funnel_text = funnel_response

        if st.session_state.funnel_text:
            st.markdown(st.session_state.funnel_text)

