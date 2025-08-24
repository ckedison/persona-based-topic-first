import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io

# --- 頁面設定 ---
st.set_page_config(
    page_title="互動式策略儀表板",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 核心功能函式 ---

@st.cache_data
def generate_embeddings(_df, api_key):
    """為 Persona DataFrame 生成 Embeddings 並快取"""
    try:
        genai.configure(api_key=api_key)
        _df['embedding_text'] = _df['summary'].fillna('') + ' | ' + \
                               _df['goals'].fillna('') + ' | ' + \
                               _df['pain_points'].fillna('') + ' | ' + \
                               _df['keywords'].fillna('')
        
        texts_to_embed = _df['embedding_text'].tolist()
        
        result = genai.embed_content(
            model='models/text-embedding-004',
            content=texts_to_embed,
            task_type="RETRIEVAL_DOCUMENT"
        )
        _df['embeddings'] = result['embedding']
        return _df
    except Exception as e:
        st.error(f"生成 Persona Embeddings 時發生錯誤: {e}")
        return None

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
    if query_fan_out_df is not None and not query_fan_out_df.empty:
        query_fan_out_section = f"""
另外，請務必參考以下由 SEO 專家分析的「Query Fan Out」資料，這代表了用戶在搜尋此主題時的真實意圖與變化：
```
{query_fan_out_df.to_markdown(index=False)}
```
"""

    return f"""
請扮演一位頂尖的內容策略顧問，擁有敏銳的用戶洞察力。
我的核心主題是：「{topic}」。

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
(請提供 3-5 個**緊扣上述「連結分析」**並**回應「Query Fan Out」意圖**的具體內容點子。每一個點子都必須包含「主題/標題方向」、「建議格式」和「理由」。)

* **點子一：**
    * **主題/標題方向:** [一個能直接反映「連結分析」的具體標題]
    * **建議格式:** [從 Persona 偏好格式中挑選]
    * **理由:** [說明為什麼這個點子和格式能有效解決 Persona 在此主題下的特定問題]

* **點子二：**
    * **主題/標題方向:** [一個能直接反映「連結分析」的具體標題]
    * **建議格式:** [從 Persona 偏好格式中挑選]
    * **理由:** [說明為什麼這個點子和格式能有效解決 Persona 在此主題下的特定問題]

請確保所有產出的點子都**高度聚焦**在核心主題與 Persona 需求的交集上，避免提出泛泛之論。
"""


def create_funnel_prompt(topic, strategy_text, query_fan_out_df=None):
    """根據初步策略生成行銷漏斗策略的 Prompt"""
    query_fan_out_section = ""
    if query_fan_out_df is not None and not query_fan_out_df.empty:
        query_fan_out_section = f"""
在規劃時，請優先考慮以下「Query Fan Out」資料中，具有高商業意圖或能解決深度問題的查詢，將其融入你的漏斗策略中：
```
{query_fan_out_df.to_markdown(index=False)}
```
"""

    return f"""
請扮演一位頂尖的數位行銷策略總監 (Head of Digital Strategy)，專精於設計高轉換率的內容行銷漏斗。
我的核心主題是：「{topic}」。
{query_fan_out_section}

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
   - **導購與行動呼籲 (CTA) 設計:** [設計強而有力的 CTA。例如：「立即訂閱親子天下Premium，解鎖超過1000篇專家文章與線上課程！」、「使用折扣碼『FUNNEL20』享首月8折優惠。」、「點此查看其他家長的真實使用心得。」]

---

**📊 總結：用戶旅程地圖**
(請用流程圖的方式，總結從 TOFU 到 BOFU 的轉換路徑)
* **[認知內容]** (例如: IG Reels 短影音) → **CTA:** "留言+1索取完整指南"
* → **[考慮內容]** (例如: 私訊發送 PDF 指南) → **CTA:** "指南中附有專屬訂閱優惠連結"
* → **[轉換內容]** (例如: 優惠訂閱頁面) → **最終目標:** 完成訂閱
"""


# --- 初始化 Session State ---
if 'persona_df' not in st.session_state:
    st.session_state.persona_df = None
if 'query_fan_out_df' not in st.session_state:
    st.session_state.query_fan_out_df = None
if 'matched_personas' not in st.session_state:
    st.session_state.matched_personas = None
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False
if 'strategy_text' not in st.session_state:
    st.session_state.strategy_text = None


# --- Streamlit 介面佈局 ---

st.title("🎯 互動式策略儀表板 (語意分析版)")
st.markdown("上傳您的 Persona，讓 AI 理解語意並為您打造主題優先的內容策略")

with st.sidebar:
    st.header("⚙️ 設定面板")

    api_key = st.text_input("請輸入您的 Gemini API 金鑰", type="password", help="[點此取得您的 API 金鑰](https://aistudio.google.com/app/apikey)")

    if api_key:
        try:
            genai.configure(api_key=api_key)
            st.session_state.api_key_configured = True
            st.info("API 金鑰已設定。")
        except Exception as e:
            st.error(f"API 金鑰設定失敗: {e}")
            st.session_state.api_key_configured = False

    st.markdown("---")

    st.subheader("1. 上傳 Persona 資料庫")
    uploaded_persona_file = st.file_uploader(
        "請上傳 Persona CSV 檔案",
        type="csv",
        key="persona_uploader",
        help="檔案需包含 `persona_name`, `summary`, `goals`, `pain_points`, `keywords`, `preferred_formats` 欄位。"
    )

    if uploaded_persona_file is not None:
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
                
                if st.session_state.api_key_configured:
                    with st.spinner("正在為 Persona 資料建立語意索引..."):
                        st.session_state.persona_df = generate_embeddings(st.session_state.persona_df, api_key)
                        if st.session_state.persona_df is not None:
                             st.info("Persona 語意索引建立完成！")
                else:
                    st.warning("請先輸入有效的 API 金鑰以建立 Persona 語意索引。")

        except Exception as e:
            st.error(f"Persona 檔案讀取失敗：{e}")
            st.session_state.persona_df = None
    
    st.markdown("---")

    st.subheader("2. (選填) 上傳 Query Fan Out")
    uploaded_query_file = st.file_uploader(
        "請上傳 Query Fan Out CSV 檔案",
        type="csv",
        key="query_uploader",
        help="檔案需包含 `query`, `type`, `user_intent`, `reasoning` 欄位。"
    )

    if uploaded_query_file is not None:
        try:
            df = pd.read_csv(uploaded_query_file)
            required_headers = ['query', 'type', 'user_intent', 'reasoning']
            missing_headers = [h for h in required_headers if h not in df.columns]

            if missing_headers:
                st.error(f"Query Fan Out CSV 檔案缺少欄位: {', '.join(missing_headers)}")
                st.session_state.query_fan_out_df = None
            else:
                st.session_state.query_fan_out_df = df
                st.success(f"成功載入 {len(df)} 筆 Query Fan Out 資料！")
        except Exception as e:
            st.error(f"Query Fan Out 檔案讀取失敗：{e}")
            st.session_state.query_fan_out_df = None


    st.markdown("---")

    st.subheader("3. 輸入核心主題")
    topic = st.text_input("輸入您想規劃內容的核心主題", placeholder="例如：青少年理財教育")

    if st.button("🔍 語意匹配 Persona", use_container_width=True, type="primary"):
        if not st.session_state.api_key_configured:
            st.warning("請先輸入並驗證您的 API 金鑰。")
        elif st.session_state.persona_df is None or 'embeddings' not in st.session_state.persona_df.columns:
            st.warning("請先上傳 Persona 資料庫並等待語意索引建立完成。")
        elif not topic:
            st.warning("請輸入核心主題。")
        else:
            with st.spinner("正在進行語意分析與匹配..."):
                try:
                    # 建立用於匹配的上下文
                    context_text = topic
                    if st.session_state.query_fan_out_df is not None:
                        queries = " ".join(st.session_state.query_fan_out_df['query'].fillna(''))
                        intents = " ".join(st.session_state.query_fan_out_df['user_intent'].fillna(''))
                        context_text = f"{topic} - 相關查詢與意圖: {queries} {intents}"

                    context_embedding_result = genai.embed_content(
                        model='models/text-embedding-004',
                        content=context_text,
                        task_type="RETRIEVAL_QUERY"
                    )
                    context_embedding = np.array(context_embedding_result['embedding']).reshape(1, -1)
                    
                    persona_embeddings = np.array(st.session_state.persona_df['embeddings'].tolist())
                    similarities = cosine_similarity(context_embedding, persona_embeddings)[0]
                    
                    df = st.session_state.persona_df.copy()
                    df['score'] = similarities
                    matched = df[df['score'] > 0.5].sort_values(by='score', ascending=False)
                    if len(matched) < 10 and len(df) > 10:
                        matched = df.sort_values(by='score', ascending=False).head(10)
                    elif len(matched) == 0:
                         matched = df.sort_values(by='score', ascending=False).head(5)


                    st.session_state.matched_personas = matched
                    st.session_state.strategy_text = None 
                except Exception as e:
                    st.error(f"語意匹配時發生錯誤: {e}")

# 主畫面
if st.session_state.matched_personas is not None:
    st.markdown("---")
    st.subheader("4. 選擇相關 Persona")
    st.markdown("以下是根據您的主題與 Query Fan Out (若有) **語意關聯度**匹配出的 Persona。")

    selected_indices = []
    
    for index, row in st.session_state.matched_personas.iterrows():
        cols = st.columns([0.1, 0.7, 0.2])
        with cols[0]:
            is_selected = st.checkbox("", key=f"persona_{index}")
            if is_selected:
                selected_indices.append(index)
        with cols[1]:
            st.markdown(f"**{row['persona_name']}**")
            st.caption(row['summary'])
        with cols[2]:
            st.info(f"關聯度: {row['score']:.0%}")

    if selected_indices:
        st.markdown("---")
        if st.button("🚀 為選定對象生成初步策略", use_container_width=True):
            if not st.session_state.api_key_configured:
                st.error("請在左側側邊欄輸入您的 Gemini API 金鑰。")
            else:
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash-latest')
                    selected_df = st.session_state.matched_personas.loc[selected_indices]
                    prompt = create_dynamic_prompt(topic, selected_df, st.session_state.query_fan_out_df)

                    with st.spinner("🧠 AI 內容顧問正在生成初步點子..."):
                        response = model.generate_content(prompt)
                        st.session_state.strategy_text = response.text

                except Exception as e:
                    st.error(f"生成初步策略時發生錯誤：{e}")
                    st.session_state.strategy_text = None

    if st.session_state.strategy_text:
        st.markdown("---")
        st.subheader("5. AI 生成的初步內容策略")
        st.markdown(st.session_state.strategy_text)

        st.markdown("---")
        st.subheader("6. 整合行銷漏斗策略")
        if st.button("🧠 生成整合行銷漏斗策略", use_container_width=True, type="primary"):
            if not st.session_state.api_key_configured:
                st.error("請在左側側邊欄輸入您的 Gemini API 金鑰。")
            else:
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash-latest')
                    funnel_prompt = create_funnel_prompt(topic, st.session_state.strategy_text, st.session_state.query_fan_out_df)
                    
                    with st.spinner("👑 AI 行銷總監正在建構漏斗策略..."):
                        funnel_response = model.generate_content(funnel_prompt)
                        st.markdown(funnel_response.text)

                except Exception as e:
                    st.error(f"生成行銷漏斗時發生錯誤：{e}")

else:
    st.info("請在左側面板完成設定，匹配結果將顯示於此。")
