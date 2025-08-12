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
        # 組合用於生成 embedding 的文字
        _df['embedding_text'] = _df['summary'].fillna('') + ' | ' + \
                               _df['goals'].fillna('') + ' | ' + \
                               _df['pain_points'].fillna('') + ' | ' + \
                               _df['keywords'].fillna('')
        
        texts_to_embed = _df['embedding_text'].tolist()
        
        # 使用 text-embedding-004 模型
        result = genai.embed_content(
            model='models/text-embedding-004', # <-- 已修正
            content=texts_to_embed,
            task_type="RETRIEVAL_DOCUMENT"
        )
        _df['embeddings'] = result['embedding']
        return _df
    except Exception as e:
        st.error(f"生成 Persona Embeddings 時發生錯誤: {e}")
        return None

def create_dynamic_prompt(topic, selected_personas_df):
    """根據主題和選擇的 Persona 動態生成 Prompt"""
    persona_details = ""
    for index, row in selected_personas_df.iterrows():
        persona_details += f"""
### 人物誌 (Persona): {row['persona_name']}
- **核心摘要:** {row.get('summary', '無')}
- **主要目標:** {row.get('goals', '無')}
- **主要痛點:** {row.get('pain_points', '無')}
- **偏好內容格式:** {row.get('preferred_formats', '無')}
"""

    return f"""
請扮演一位頂尖的內容策略顧問。
我的核心主題是：「{topic}」。

我需要針對以下的人物誌 (Persona) 規劃一份全面且可執行的內容策略。請務必根據每個人物誌的獨特痛點、目標和「偏好內容格式」來客製化策略。

這是我要你分析的人物誌資料：
{persona_details}

請為 **每一個** 人物誌提供一份獨立的策略建議，並遵循以下格式輸出，使用 Markdown 語法：

---

### **針對「[人物誌姓名]」的內容策略**

**1. 核心策略角度:**
(請用一句話總結針對此 Persona 的溝通核心，例如：強調數據驅動的 ROI 提升，而非空泛的品牌故事。)

**2. 內容支柱與格式建議 (Content Pillars & Formats):**
(請提供 5-8 個具體的內容點子。每一個點子都要包含「主題/標題方向」和「建議格式」。格式必須從該 Persona 的偏好格式中挑選，並說明選擇此格式的原因。)

* **點子一：**
    * **主題/標題方向:** [具體的標題方向]
    * **建議格式:** [例如：YouTube 深度影片]
    * **理由:** [為什麼這個格式適合這個點子與 Persona]

* **點子二：**
    * **主題/標題方向:** [具體的標題方向]
    * **建議格式:** [例如：研究報告 (PDF)]
    * **理由:** [為什麼這個格式適合這個點子與 Persona]

* **點子三：**
    * **主題/標題方向:** [具體的標題方向]
    * **建議格式:** [例如：TikTok 短影音腳本點子]
    * **理由:** [為什麼這個格式適合這個點子與 Persona]

請確保你的建議具體、有創意且高度相關。
"""

# --- 初始化 Session State ---
if 'persona_df' not in st.session_state:
    st.session_state.persona_df = None
if 'matched_personas' not in st.session_state:
    st.session_state.matched_personas = None
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False

# --- Streamlit 介面佈局 ---

# 標題
st.title("🎯 互動式策略儀表板 (語意分析版)")
st.markdown("上傳您的 Persona，讓 AI 理解語意並為您打造主題優先的內容策略")

# 側邊欄 (控制面板)
with st.sidebar:
    st.header("⚙️ 設定面板")

    # API 金鑰輸入
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

    # 步驟一：上傳檔案
    st.subheader("1. 上傳 Persona 資料庫")
    uploaded_file = st.file_uploader(
        "請上傳 CSV 檔案",
        type="csv",
        help="檔案需包含 `persona_name`, `summary`, `goals`, `pain_points`, `keywords`, `preferred_formats` 欄位。"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_headers = ['persona_name', 'summary', 'goals', 'pain_points', 'keywords', 'preferred_formats']
            missing_headers = [h for h in required_headers if h not in df.columns]

            if missing_headers:
                st.error(f"CSV 檔案缺少必要欄位: {', '.join(missing_headers)}")
                st.session_state.persona_df = None
            else:
                st.session_state.persona_df = df
                st.success(f"成功載入 {len(df)} 筆 Persona 資料！")
                
                # 在此預先生成 Embeddings
                if st.session_state.api_key_configured:
                    with st.spinner("正在為 Persona 資料建立語意索引..."):
                        st.session_state.persona_df = generate_embeddings(st.session_state.persona_df, api_key)
                        if st.session_state.persona_df is not None:
                             st.info("語意索引建立完成！")
                else:
                    st.warning("請先輸入有效的 API 金鑰以建立語意索引。")

        except Exception as e:
            st.error(f"檔案讀取失敗：{e}")
            st.session_state.persona_df = None

    st.markdown("---")

    # 步驟二：輸入主題
    st.subheader("2. 輸入核心主題")
    topic = st.text_input("輸入您想規劃內容的核心主題", placeholder="例如：青少年理財教育")

    # 匹配按鈕
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
                    # 為主題生成 Embedding
                    topic_embedding_result = genai.embed_content(
                        model='models/text-embedding-004', # <-- 已修正
                        content=topic,
                        task_type="RETRIEVAL_QUERY"
                    )
                    topic_embedding = np.array(topic_embedding_result['embedding']).reshape(1, -1)
                    
                    # 計算餘弦相似度
                    persona_embeddings = np.array(st.session_state.persona_df['embeddings'].tolist())
                    similarities = cosine_similarity(topic_embedding, persona_embeddings)[0]
                    
                    # 更新 DataFrame 並排序
                    df = st.session_state.persona_df.copy()
                    df['score'] = similarities
                    # 顯示分數大於 0.5 的結果，或至少顯示前10名
                    matched = df[df['score'] > 0.5].sort_values(by='score', ascending=False)
                    if len(matched) < 10:
                        matched = df.sort_values(by='score', ascending=False).head(10)

                    st.session_state.matched_personas = matched
                except Exception as e:
                    st.error(f"語意匹配時發生錯誤: {e}")


# 主畫面 (結果顯示)
if st.session_state.matched_personas is not None:
    st.markdown("---")
    st.subheader("3. 選擇相關 Persona")
    st.markdown("以下是根據您的主題**語意關聯度**匹配出的 Persona，請勾選您想為其規劃策略的對象。")

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
        if st.button("🚀 為選定對象生成策略", use_container_width=True):
            if not st.session_state.api_key_configured:
                st.error("請在左側側邊欄輸入您的 Gemini API 金鑰。")
            else:
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash-latest')
                    selected_df = st.session_state.matched_personas.loc[selected_indices]
                    prompt = create_dynamic_prompt(topic, selected_df)

                    with st.spinner("🧠 AI 策略師正在為您撰寫策略，請稍候..."):
                        response = model.generate_content(prompt)
                        st.subheader("4. AI 生成的內容策略")
                        st.markdown(response.text)

                except Exception as e:
                    st.error(f"生成策略時發生錯誤：{e}")
else:
    st.info("請在左側面板完成設定，匹配結果將顯示於此。")
