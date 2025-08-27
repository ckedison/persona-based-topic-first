# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io
import re
import ast

# --- 頁面設定 ---
st.set_page_config(
    page_title="Topic first 內容策略產生器",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 隱藏 Streamlit UI 元件 ---
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- 核心功能函式 (Prompt Engineering & API Calls) ---

def create_iterative_persona_prompt(topic):
    """建立一個指導使用者進行迭代生成的 Persona Prompt"""
    return f"""
請扮演一位頂尖的市場研究與用戶體驗專家。
我的核心產品/服務主題是：「{topic}」。

**你的任務**
你的任務是為這個主題生成 10 個**與主題直接相關、且極具代表性**的潛在目標人物誌 (Persona)。為了確保最高品質，請嚴格遵循以下的思考與執行流程：

**思考流程 (Chain of Thought):**

1.  **第一步：解構主題**
    * 首先，請在內心分析「{topic}」這個主題的核心價值是什麼？它主要解決了誰的什麼問題？

2.  **第二步：發想核心用戶群**
    * 基於上述分析，腦力激盪出 5-7 個最可能對此主題有強烈需求的**具體用戶群體** (例如，如果主題是「幼兒情緒管理」，用戶群可能是「家有2-5歲幼兒的新手父母」、「幼兒園老師」、「兒童心理諮商師」等)。

3.  **第三步：從用戶群到 Persona**
    * 從你發想的用戶群中，挑選並深化，創造出 10 個具體的 Persona。每一個 Persona 都必須有一個清晰的背景故事，讓他/她與「{topic}」的連結不言而喻。**避免生成「學生」、「上班族」等過於模糊的角色。**

**最終輸出**
在完成上述思考流程後，請**只**提供最終的 10 份 Persona。

**輸出格式**
請嚴格遵循以下 CSV 格式，包含標頭，不要有任何步驟描述或其他文字。每一筆資料的欄位內容請用雙引號 `"` 包覆。

```csv
"persona_name","summary","goals","pain_points","keywords","preferred_formats"
"範例人物誌1","範例摘要1","範例目標1","範例痛點1","關鍵字1,關鍵字2","格式1,格式2"
... (直到第10筆)
```

**生成指南 (極度重要):**
- **persona_name:** 給一個具體且有代表性的名字 (例如: 焦慮的新手媽媽 怡君)。
- **summary:** 一句話總結這個 Persona 的核心特徵，**並明確點出他與「{topic}」的直接關係**。
- **goals:** 他們在「{topic}」這個主題上，最想達成的 2-3 個**具體**目標。
- **pain_points:** 他們在「{topic}」這個主題上，遇到的 2-3 個**真實且迫切**的困難或煩惱。
- **keywords:** 他們為了解決上述痛點或達成目標時，**最可能**用來搜尋的 3-5 個關鍵字。
- **preferred_formats:** 他們最喜歡用來接收**與「{topic}」相關資訊**的 3-4 種內容格式。

請開始執行。
"""

def create_embedding_script(df_string, api_key):
    """生成本地執行的 Python 腳本以建立 Embeddings"""
    return f"""
# -*- coding: utf-8 -*-
import pandas as pd
import google.generativeai as genai
import io
import time

# --- 設定 ---
# !!! 重要 !!! 請在此處貼上您自己的 Gemini API 金鑰
API_KEY = "{api_key or '請在此處貼上您的API金鑰'}"
CSV_DATA = '''
{df_string}
'''
OUTPUT_FILENAME = "personas_with_embeddings.csv"

# --- 主程式 ---
def main():
    print("1. 正在設定 API 金鑰...")
    if not API_KEY or "請在此處" in API_KEY:
        print("錯誤：API 金鑰為空，請在腳本中填寫您的 Gemini API 金鑰。")
        return
    try:
        genai.configure(api_key=API_KEY)
    except Exception as e:
        print(f"錯誤：API 金鑰設定失敗 - {{e}}")
        return

    print("2. 正在讀取 Persona 資料...")
    try:
        csv_io = io.StringIO(CSV_DATA)
        df = pd.read_csv(csv_io)
        print(f"成功讀取 {{len(df)}} 筆 Persona。")
    except Exception as e:
        print(f"錯誤：CSV 資料讀取失敗 - {{e}}")
        return

    print("3. 正在準備文字以進行語意分析...")
    df['embedding_text'] = df['summary'].fillna('') + ' | ' + \\
                           df['goals'].fillna('') + ' | ' + \\
                           df['pain_points'].fillna('') + ' | ' + \\
                           df['keywords'].fillna('')
    
    texts_to_embed = df['embedding_text'].tolist()

    print(f"4. 正在為 {{len(texts_to_embed)}} 筆資料請求語意向量 (Embeddings)...")
    print("   (這個步驟可能會需要一些時間，且會消耗您的 API 配額)")
    
    try:
        result = genai.embed_content(
            model='models/text-embedding-004',
            content=texts_to_embed,
            task_type="RETRIEVAL_DOCUMENT"
        )
        df['embeddings'] = result['embedding']
        print("   語意向量生成成功！")
    except Exception as e:
        print(f"錯誤：語意向量生成失敗 - {{e}}")
        print("   請檢查您的 API 金鑰與方案配額。")
        return

    print(f"5. 正在將結果儲存至檔案: {{OUTPUT_FILENAME}}")
    # 將 list 轉換為 string 以便儲存
    df['embeddings'] = df['embeddings'].apply(str)
    df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8-sig')
    
    print("\\n---")
    print("✅ 任務完成！")
    print(f"請將生成的檔案 '{{OUTPUT_FILENAME}}' 上傳回 Streamlit 應用程式中。")

if __name__ == "__main__":
    main()
"""


def create_query_fan_out_prompt(topic):
    """為 AI 生成 Query Fan Out 建立 Prompt"""
    return f"""
請扮演一位資深的 SEO 與內容策略專家。
我的核心主題是：「{topic}」。

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

def generate_query_fan_out_with_gemini(topic, api_key):
    """使用 Gemini API 生成 Query Fan Out DataFrame"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = create_query_fan_out_prompt(topic)
        response = model.generate_content(prompt)
        
        csv_text = response.text.strip().replace('```csv', '').replace('```', '')
        
        csv_io = io.StringIO(csv_text)
        df = pd.read_csv(csv_io)
        
        required_headers = ['query', 'type', 'user_intent', 'reasoning']
        if not all(h in df.columns for h in required_headers):
            st.error("AI 生成的 Query Fan Out 格式不符，請稍後再試。")
            return None
            
        return df
    except Exception as e:
        st.error(f"自動生成 Query Fan Out 時發生錯誤: {e}")
        return None

def process_and_embed_personas(df, api_key):
    """為 Persona DataFrame 生成 Embeddings"""
    try:
        genai.configure(api_key=api_key)
        df['embedding_text'] = df['summary'].fillna('') + ' | ' + \
                               df['goals'].fillna('') + ' | ' + \
                               df['pain_points'].fillna('') + ' | ' + \
                               df['keywords'].fillna('')
        
        texts_to_embed = df['embedding_text'].tolist()
        
        result = genai.embed_content(
            model='models/text-embedding-004',
            content=texts_to_embed,
            task_type="RETRIEVAL_DOCUMENT"
        )
        df['embeddings'] = result['embedding']
        return df
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

* **點子二：**
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

* **點子二：**
    * **主題/標題方向:** [一個能直接反映「連結分析」的具體標題]
    * **建議格式:** [從 Persona 偏好格式中挑選]
    * **理由:** [說明為什麼這個點子和格式能有效解決 Persona 在此主題下的特定問題]
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
{idea_format_instruction}
{idea_structure}

請確保所有產出的點子都**高度聚焦**在核心主題與 Persona 需求的交集上，避免提出泛泛之論。

---

### **總結：內容產製清單 (Content Production Checklist)**

現在，請扮演一位**內容製作總監**。請回顧以上**所有**為不同 Persona 生成的內容點子，並將它們整合成一個清晰的總表。

這個表格的目的是讓團隊一目而然地知道總共需要製作哪些類型的內容，以及每個類型有哪些具體的點子。

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

st.title("🚀 Topic first 內容策略產生器 (beta)")
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

    st.subheader("1. 輸入核心主題")
    topic = st.text_input("輸入您想規劃內容的核心主題", placeholder="例如：青少年理財教育")

    st.markdown("---")

    st.subheader("2. Persona 資料")
    
    # 區塊 A: AI 輔助生成
    with st.expander("AI 輔助生成 Persona (建議)"):
        st.markdown("若您沒有現成的 Persona 檔案，可使用此功能。")
        
        if st.button("產生 Persona 生成指令", key="gen_persona_prompt"):
            if not topic:
                st.warning("請先輸入核心主題。")
            else:
                st.session_state.persona_prompt = create_iterative_persona_prompt(topic)

        if 'persona_prompt' in st.session_state:
            st.text_area("1. 複製以下指令，並到您的 Gemini 介面執行", value=st.session_state.persona_prompt, height=200)
            
            pasted_persona_csv = st.text_area("2. 將 Gemini 生成的 CSV 結果貼於此處", height=150, key="pasted_persona")
            
            if st.button("處理貼上的 Persona 資料", key="process_pasted_persona"):
                if pasted_persona_csv:
                    try:
                        # 智慧解析貼上的內容
                        match = re.search(r'```csv\n(.*?)\n```', pasted_persona_csv, re.DOTALL)
                        if match:
                            csv_text = match.group(1)
                        else:
                            required_headers = ['persona_name', 'summary', 'goals', 'pain_points', 'keywords', 'preferred_formats']
                            header_str = '"' + '","'.join(required_headers) + '"'
                            csv_start_index = pasted_persona_csv.find(header_str)
                            if csv_start_index != -1:
                                csv_text = pasted_persona_csv[csv_start_index:]
                            else:
                                csv_text = pasted_persona_csv

                        csv_io = io.StringIO(csv_text)
                        df = pd.read_csv(csv_io)
                        st.session_state.persona_df = df
                        st.success(f"成功處理 {len(df)} 筆貼上的 Persona 資料！")
                    except Exception as e:
                        st.error(f"處理貼上資料時發生錯誤，請確認格式是否為標準 CSV: {e}")
                else:
                    st.warning("請先貼上資料。")

    # 區塊 B: 上傳檔案
    uploaded_persona_file = st.file_uploader(
        "或上傳您自己的 Persona CSV 檔案",
        type="csv",
        key="persona_uploader",
    )
    if uploaded_persona_file:
        try:
            # 將上傳的檔案轉換為 DataFrame
            df = pd.read_csv(uploaded_persona_file)
            
            # 檢查必要的欄位是否存在
            required_headers = ['persona_name', 'summary', 'goals', 'pain_points', 'keywords', 'preferred_formats']
            
            if 'embeddings' in df.columns:
                required_headers.append('embeddings')

            missing_headers = [h for h in required_headers if h not in df.columns]

            if 'embeddings' not in df.columns:
                 st.warning("提醒：您上傳的檔案不含語意向量 (Embeddings)。")

            if missing_headers and 'embeddings' not in missing_headers:
                st.error(f"Persona CSV 檔案缺少欄位: {', '.join(missing_headers)}")
                st.session_state.persona_df = None
            else:
                st.session_state.persona_df = df
                st.success(f"成功載入 {len(df)} 筆 Persona 資料！")
        except Exception as e:
            st.error(f"Persona 檔案讀取失敗：{e}")
            st.session_state.persona_df = None
    
    # 區塊 C: 建立語意索引 (選填)
    if st.session_state.persona_df is not None and 'embeddings' not in st.session_state.persona_df.columns:
        st.markdown("---")
        st.subheader("建立語意索引 (選填)")
        st.markdown("點擊下方按鈕，可為您的 Persona 資料建立語意索引，以提升匹配精準度。")
        
        if st.button("在 App 中建立索引", key="embed_in_app"):
            if not st.session_state.api_key_configured:
                st.warning("請先輸入 API 金鑰。")
            else:
                with st.spinner("正在為 Persona 資料建立語意索引..."):
                    st.session_state.persona_df = process_and_embed_personas(st.session_state.persona_df, api_key)
                    if st.session_state.persona_df is not None:
                        st.success("語意索引建立完成！")

        with st.expander("或產生本地端執行腳本 (推薦)"):
            st.markdown("若資料量龐大，建議產生 Python 腳本在您自己的電腦上執行，以避免 API 超額問題。")
            if st.button("產生本地端執行腳本", key="gen_embedding_script"):
                df_string = st.session_state.persona_df.to_csv(index=False)
                st.session_state.embedding_script = create_embedding_script(df_string, api_key)
            
            if 'embedding_script' in st.session_state:
                st.text_area("1. 複製以下 Python 程式碼，儲存成 .py 檔案", value=st.session_state.embedding_script, height=200)
                st.markdown("2. 在您的電腦上安裝必要的套件 (`pip install pandas google-generativeai`) 並執行此腳本。")
                st.markdown("3. 執行成功後，將生成的 `personas_with_embeddings.csv` 檔案，透過上方的上傳區塊重新上傳。")


    st.markdown("---")

    st.subheader("3. Query Fan Out 資料 (選填)")
    uploaded_query_file = st.file_uploader(
        "上傳 Query Fan Out CSV 檔案",
        type="csv",
        key="query_uploader",
    )
    if uploaded_query_file:
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
            
    if uploaded_query_file is None and st.session_state.query_fan_out_df is None:
        if st.button("📊 自動生成 Query Fan Out", use_container_width=True):
            if not st.session_state.api_key_configured or not topic:
                st.warning("請先輸入 API 金鑰和核心主題。")
            else:
                with st.spinner("正在為您自動生成相關查詢..."):
                    generated_qfo_df = generate_query_fan_out_with_gemini(topic, api_key)
                    if generated_qfo_df is not None:
                        st.session_state.query_fan_out_df = generated_qfo_df
                        st.success(f"已成功為您生成 {len(generated_qfo_df)} 筆相關查詢！")

    st.markdown("---")

    if st.button("🔍 執行策略分析", use_container_width=True, type="primary"):
        if not st.session_state.api_key_configured:
            st.warning("請先輸入並驗證您的 API 金鑰。")
        elif not topic:
            st.warning("請輸入核心主題。")
        elif st.session_state.persona_df is None:
            st.warning("請先上傳或生成並處理 Persona 資料。")
        else:
            # 執行匹配
            with st.spinner("正在進行分析與匹配..."):
                try:
                    df = st.session_state.persona_df.copy()
                    
                    # 判斷使用何種匹配模式
                    if 'embeddings' in df.columns and not df['embeddings'].isnull().all():
                        st.info("偵測到語意索引，將使用語意分析模式。")
                        # 將儲存為字串的 embeddings 轉回 list of floats
                        df['embeddings'] = df['embeddings'].apply(ast.literal_eval)

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
                        
                        persona_embeddings = np.array(df['embeddings'].tolist())
                        similarities = cosine_similarity(context_embedding, persona_embeddings)[0]
                        df['score'] = similarities
                    else:
                        st.info("未偵測到語意索引，將使用關鍵字匹配模式。")
                        context_text = topic
                        if st.session_state.query_fan_out_df is not None:
                            queries = " ".join(st.session_state.query_fan_out_df['query'].fillna(''))
                            context_text += " " + queries
                        
                        topic_tokens = set(context_text.lower().split())

                        def calculate_keyword_score(row):
                            score = 0
                            searchable_text = f"{row.get('pain_points', '')} {row.get('keywords', '')}".lower()
                            for token in topic_tokens:
                                if token in searchable_text:
                                    score += 1
                            return score
                        
                        df['score'] = df.apply(calculate_keyword_score, axis=1)

                    matched = df.sort_values(by='score', ascending=False).head(10)
                    st.session_state.matched_personas = matched
                    st.session_state.strategy_text = None 
                except Exception as e:
                    st.error(f"策略分析時發生錯誤: {e}")

# 主畫面
if st.session_state.matched_personas is not None:
    st.markdown("---")
    st.subheader("4. 選擇相關 Persona")
    st.markdown("以下是根據您的主題與 Query Fan Out (若有) **語意關聯度**匹配出的 Persona。")

    selected_indices = []
    
    if st.session_state.matched_personas.empty:
        st.warning("找不到符合條件的 Persona，請嘗試調整核心主題或檢查上傳的檔案。")
    else:
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
                st.info(f"關聯度: {row['score']:.0%}" if isinstance(row['score'], float) else f"分數: {row['score']}")

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
        
        with st.form(key='funnel_form'):
            st.markdown("**在生成最終漏斗前，請設定您的轉換目標：**")
            
            product_name = st.text_input("產品/服務名稱", placeholder="例如：親子理財線上課")
            conversion_action = st.selectbox("期望轉換動作", 
                                             ['購買商品', '填寫表單', '預約諮詢', '訂閱服務', '下載App'])
            target_url = st.text_input("目標網址 (URL)", placeholder="https://example.com/product-page")
            product_desc = st.text_area("產品/服務簡介 (選填)", placeholder="簡要說明您的產品特色與價值")

            submit_button = st.form_submit_button(label="🧠 生成整合行銷漏斗策略", use_container_width=True, type="primary")

            if submit_button:
                if not st.session_state.api_key_configured:
                    st.error("請在左側側邊欄輸入您的 Gemini API 金鑰。")
                elif not product_name or not target_url:
                    st.warning("請填寫「產品/服務名稱」與「目標網址」。")
                else:
                    conversion_goal = {
                        "name": product_name,
                        "action": conversion_action,
                        "url": target_url,
                        "desc": product_desc
                    }
                    try:
                        model = genai.GenerativeModel('gemini-1.5-flash-latest')
                        funnel_prompt = create_funnel_prompt(topic, st.session_state.strategy_text, conversion_goal, st.session_state.query_fan_out_df)
                        
                        with st.spinner("👑 AI 行銷總監正在建構漏斗策略..."):
                            funnel_response = model.generate_content(funnel_prompt)
                            st.markdown(funnel_response.text)

                    except Exception as e:
                        st.error(f"生成行銷漏斗時發生錯誤：{e}")

else:
    st.info("請在左側面板完成設定，匹配結果將顯示於此。")
    st.markdown("---")
    st.subheader("格式範例")
    
    st.markdown("**Persona 資料格式範例**")
    persona_example_data = {
        'persona_name': ['新手媽媽 怡君'],
        'summary': ['家有1歲嬰兒的全職媽媽，對嬰兒發展感到焦慮，渴望獲得專家指引'],
        'goals': ['希望孩子健康成長，了解各階段發展里程碑'],
        'pain_points': ['資訊過載但不知如何篩選，擔心自己做得不夠好'],
        'keywords': ['嬰兒發展,副食品,睡眠引導'],
        'preferred_formats': ['Podcast,IG圖文卡,深度文章']
    }
    st.dataframe(pd.DataFrame(persona_example_data))

    st.markdown("**Query Fan Out 資料格式範例**")
    query_example_data = {
        'query': ['如何教小孩理財'],
        'type': ['問題 (Question)'],
        'user_intent': ['尋找具體的兒童理財教育方法與步驟'],
        'reasoning': ['這是核心主題最直接的資訊型查詢']
    }
    st.dataframe(pd.DataFrame(query_example_data))


st.sidebar.markdown("---")
st.sidebar.caption("此工具由劉呈逸開發 (https://www.facebook.com/edison.liu.180)")
