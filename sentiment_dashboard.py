# -*- coding: utf-8 -*-
"""
Anket Sentiment Analizi - Streamlit Dashboard
===================================================
Anket sonuçlarının görselleştirilmesi ve analizi için interaktif dashboard
Harmanlanmış versiyon: streamlit_dashboard.py görünümü + sentiment_dashboard.py analizleri
"""

import pandas as pd
import numpy as np
import re
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
warnings.filterwarnings('ignore')

# Sayfa yapılandırması
st.set_page_config(
    page_title="Anket Analizi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral": "#95a5a6",
    "mixed": "#f39c12",
}

LIKERT_GROUPS = {
    "Genel": [1, 2, 3, 4, 5],
    "Kasko": [6, 7, 8, 9, 10, 11],
    "TSS": [13, 14, 15, 16, 17],
    "Konut": [19, 20, 21, 22, 23, 24],
    "Ticari": [26, 27, 28, 29, 30, 31],
    "Hasar": [33, 34, 35, 36],
    "Üretim Operasyon": [37, 38, 39],
    "Tahsilat": [40, 41, 42],
    "Dijital & HelpDesk": [43, 44, 45, 46, 47],
    "NPS": [48, 49],
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_sentiment_label(value: str) -> str:
    """Normalize sentiment labels to standard form."""
    if pd.isna(value) or value == "":
        return ""
    
    value_lower = str(value).strip().lower()
    
    if value_lower in ["nötral", "nötr", "neutral"]:
        return "neutral"
    elif value_lower in ["olumlu", "pozitif", "positive"]:
        return "positive"
    elif value_lower in ["olumsuz", "negatif", "negative"]:
        return "negative"
    elif value_lower in ["karma", "karışık", "karisik", "mixed"]:
        return "mixed"
    else:
        return value_lower


def sentiment_label_to_score(value: str) -> float:
    """Convert sentiment label to numeric score."""
    normalized = normalize_sentiment_label(value)
    
    if normalized == "positive":
        return 1.0
    elif normalized == "negative":
        return -1.0
    elif normalized in ["neutral", "mixed"]:
        return 0.0
    else:
        return np.nan


def extract_numeric_from_answer(value, max_valid=5, treat_ge_as_nan=True, allow_zero=False):
    """Extract numeric value from Likert scale or NPS answers.
    
    Parameters:
    -----------
    value : str or numeric
        The answer value to parse
    max_valid : int
        Maximum valid value (default 5 for Likert, 10 for NPS)
    treat_ge_as_nan : bool
        If True, values >= max_valid+1 are treated as NaN (for "Kullanmıyorum (6)" in Likert)
        If False, values are only NaN if outside 0..max_valid range
    allow_zero : bool
        If True, 0 is a valid value (for NPS)
        If False, values < 1 are NaN (for Likert)
    """
    if pd.isna(value):
        return np.nan
    
    value_str = str(value).strip()
    match = re.search(r"\((\d+)\)", value_str)
    if match:
        num = int(match.group(1))
        
        # Check zero handling
        if not allow_zero and num < 1:
            return np.nan
        
        # Check max_valid handling
        if treat_ge_as_nan:
            # For Likert: 6+ is "Kullanmıyorum" -> NaN
            if num > max_valid:
                return np.nan
        else:
            # For NPS: only invalid if outside 0..max_valid
            if num < 0 or num > max_valid:
                return np.nan
        
        # Return valid numeric value
        if allow_zero:
            if 0 <= num <= max_valid:
                return float(num)
        else:
            if 1 <= num <= max_valid:
                return float(num)
    
    return np.nan


def get_question_col_by_num(df, num):
    """Find the question column for a given question number."""
    num_str = str(num)
    pattern = f"^{num_str}- "
    
    for col in df.columns:
        col_str = str(col)
        if re.match(pattern, col_str):
            if "Sentiment" not in col_str and "Topics" not in col_str and "__sentiment" not in col_str.lower() and "__topics" not in col_str.lower():
                return col
    
    return None


def get_question_short_name(question_col):
    """Extract short name from question column."""
    if pd.isna(question_col):
        return "Bilinmeyen"
    
    col_str = str(question_col)
    match = re.match(r"^(\d+)- (.+)", col_str)
    if match:
        num = match.group(1)
        desc = match.group(2)[:50]
        return f"{num}- {desc}..."
    
    return col_str[:50] if len(col_str) > 50 else col_str


def extract_question_short_name(full_name):
    """Uzun soru isimlerini kısalt"""
    if pd.isna(full_name):
        return "Bilinmeyen"
    
    parts = str(full_name).split("-")
    if len(parts) > 1:
        num = parts[0].strip()
        desc = parts[1].strip()[:50]
        return f"{num}- {desc}..."
    return str(full_name)[:50]


def find_sentiment_column(df, question_num):
    """Find sentiment column for a question number - try multiple formats"""
    patterns = [
        f"^{question_num}- Sentiment$",
        f"^{question_num}- __sentiment$",
        f"^{question_num}- .*Sentiment",
        f"^{question_num}- .*__sentiment"
    ]
    
    for pattern in patterns:
        for col in df.columns:
            if re.match(pattern, str(col), re.IGNORECASE):
                return col
    
    # Fallback: search for any column with question number and sentiment
    for col in df.columns:
        col_str = str(col)
        if re.match(f"^{question_num}-", col_str) and ("sentiment" in col_str.lower() or "__sentiment" in col_str.lower()):
            if "__sentiment_score" not in col_str.lower() and "score" not in col_str.lower():
                return col
    
    return None


def find_topics_column(df, question_num):
    """Find topics column for a question number - try multiple formats"""
    patterns = [
        f"^{question_num}- Topics$",
        f"^{question_num}- __topics$",
        f"^{question_num}- .*Topics",
        f"^{question_num}- .*__topics"
    ]
    
    for pattern in patterns:
        for col in df.columns:
            if re.match(pattern, str(col), re.IGNORECASE):
                if "__free_topics" not in str(col).lower():
                    return col
    
    # Fallback: search for any column with question number and topics
    for col in df.columns:
        col_str = str(col)
        if re.match(f"^{question_num}-", col_str) and ("topics" in col_str.lower() or "__topics" in col_str.lower()):
            if "__free_topics" not in col_str.lower():
                return col
    
    return None


def find_free_topics_column(df, question_num):
    """Find free topics column for a question number"""
    patterns = [
        f"^{question_num}- Free Topics$",
        f"^{question_num}- __free_topics$",
        f"^{question_num}- .*Free Topics",
        f"^{question_num}- .*__free_topics"
    ]
    
    for pattern in patterns:
        for col in df.columns:
            if re.match(pattern, str(col), re.IGNORECASE):
                return col
    
    # Fallback
    for col in df.columns:
        col_str = str(col)
        if re.match(f"^{question_num}-", col_str) and "__free_topics" in col_str.lower():
            return col
    
    return None


def find_sentiment_score_column(df, question_num, sentiment_col):
    """Find existing numeric sentiment score column for a question number.
    
    Tries multiple patterns to find columns like:
    - "12- __sentiment_score"
    - "12- ... __sentiment_score"
    - "12- Sentiment__score" (old pattern)
    
    Returns the column name if found, None otherwise.
    """
    patterns = [
        f"^{question_num}- __sentiment_score$",  # Exact match: "12- __sentiment_score"
        f"^{question_num}- .*__sentiment_score$",  # With any text: "12- ... __sentiment_score"
        f"^{sentiment_col}__score$",  # Old pattern: "12- Sentiment__score" or "12- __sentiment__score"
    ]
    
    for pattern in patterns:
        for col in df.columns:
            if re.match(pattern, str(col), re.IGNORECASE):
                # Verify it's actually numeric by checking a sample value
                sample_values = df[col].dropna().head(5)
                if len(sample_values) > 0:
                    # Check if values are numeric
                    try:
                        numeric_values = pd.to_numeric(sample_values, errors='coerce')
                        if numeric_values.notna().any():
                            return col
                    except:
                        pass
    
    return None


# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

@st.cache_data
def load_data():
    """Load and process the Excel file."""
    script_dir = Path(__file__).parent
    excel_paths = [
        script_dir / "Anket_Sentiment_Hybrid.xlsx",
        script_dir.parent / "hybrid_out" / "Anket_Sentiment_Hybrid.xlsx",
        script_dir / "hybrid_out" / "Anket_Sentiment_Hybrid.xlsx"
    ]
    
    df = None
    df_original = None
    df_konular = None
    df_likert_scores = None
    excel_path = None
    
    for path in excel_paths:
        if path.exists():
            excel_path = path
            try:
                try:
                    scored_df = pd.read_excel(path, sheet_name='scored')
                    summary_df = pd.read_excel(path, sheet_name='summary')
                    df = scored_df
                    # Try to load original sheet for Likert questions
                    try:
                        df_original = pd.read_excel(path, sheet_name='original')
                    except:
                        df_original = None
                    # Try to load konular sheet for question groupings
                    try:
                        df_konular = pd.read_excel(path, sheet_name='konular')
                    except:
                        df_konular = None
                    # Try to load likert+NPS sheet
                    try:
                        df_likert_scores = pd.read_excel(path, sheet_name='likert+NPS')
                    except:
                        df_likert_scores = None
                    break
                except:
                    df = pd.read_excel(path)
                    summary_df = pd.DataFrame()
                    break
            except Exception as e:
                continue
    
    if df is None:
        return None, None
    
    # Detect open-ended questions
    open_question_nums = []
    open_question_meta = []
    known_open_nums = ["12", "18", "25", "32", "50", "51", "52", "53"]
    
    # Detect from column names
    for col in df.columns:
        col_str = str(col)
        match = re.match(r"^(\d+)-", col_str)
        if match:
            num = match.group(1)
            if find_sentiment_column(df, num):
                if num not in open_question_nums:
                    open_question_nums.append(num)
    
    # Add known numbers
    for num in known_open_nums:
        if num not in open_question_nums:
            open_question_nums.append(num)
    
    # Create metadata for open-ended questions
    for num in open_question_nums:
        question_col = get_question_col_by_num(df, num)
        sentiment_col = find_sentiment_column(df, num)
        topics_col = find_topics_column(df, num)
        
        if question_col and sentiment_col:
            question_short = get_question_short_name(question_col)
            open_question_meta.append({
                "num": num,
                "question_col": question_col,
                "question_short": question_short,
                "sentiment_col": sentiment_col,
                "topics_col": topics_col if topics_col else None
            })
    
    # Process open-ended sentiment scores - use existing __sentiment_score columns from Excel
    sentiment_score_cols = []
    
    for meta in open_question_meta:
        sentiment_col = meta["sentiment_col"]
        question_num = meta["num"]
        
        if sentiment_col in df.columns:
            # Try to find existing numeric score column using flexible patterns
            existing_score_col = find_sentiment_score_column(df, question_num, sentiment_col)
            
            if existing_score_col:
                # Use existing numeric score column from Excel
                meta["sentiment_score_col"] = existing_score_col
            else:
                # Fallback: create from label if score column doesn't exist
                score_col = f"{sentiment_col}__score"
                df[score_col] = df[sentiment_col].apply(sentiment_label_to_score)
                meta["sentiment_score_col"] = score_col
            
            sentiment_score_cols.append(meta["sentiment_score_col"])
    
    # Calculate overall open sentiment average
    if sentiment_score_cols:
        df["OpenSentiment_Avg"] = df[sentiment_score_cols].mean(axis=1, skipna=True)
    
    # Create thematic open sentiment columns
    # Note: This is done after open_question_meta is fully populated with sentiment_score_col
    # We'll do this later after open_question_meta is complete
    
    # Add free_topics_col to metadata
    for meta in open_question_meta:
        question_num = meta["num"]
        free_topics_col = find_free_topics_column(df, question_num)
        meta["free_topics_col"] = free_topics_col if free_topics_col else None
    
    # Create open_long master table
    open_long_rows = []
    df_reset = df.reset_index()
    
    for meta in open_question_meta:
        question_col = meta["question_col"]
        sentiment_col = meta["sentiment_col"]
        topics_col = meta.get("topics_col")
        question_num = meta["num"]
        question_short = meta.get("question_short", f"Soru {question_num}")
        
        if question_col not in df.columns or sentiment_col not in df.columns:
            continue
        
        # Filter rows with non-empty text
        mask = df[question_col].notna() & (df[question_col].astype(str).str.strip() != "")
        relevant_rows = df_reset[mask].copy()
        
        if relevant_rows.empty:
                    continue
                
        # Get sentiment_score_col from metadata
        sentiment_score_col = meta.get("sentiment_score_col")
        
        # Process each row
        for idx, row in relevant_rows.iterrows():
            response_index = row.get("index", idx)
            text = str(row[question_col]).strip()
            sentiment_raw = row[sentiment_col] if pd.notna(row[sentiment_col]) else ""
            sentiment = normalize_sentiment_label(str(sentiment_raw))
            # Use existing numeric sentiment_score from Excel if available
            if sentiment_score_col and sentiment_score_col in df.columns:
                sentiment_score = row[sentiment_score_col] if pd.notna(row[sentiment_score_col]) else np.nan
            else:
                # Fallback to label-based score
                sentiment_score = sentiment_label_to_score(str(sentiment_raw))
            topics_raw = str(row[topics_col]) if topics_col and pd.notna(row[topics_col]) else ""
            
            # Explode topics
            if topics_raw and topics_raw != "nan" and topics_raw.strip():
                # Normalize topics: lowercase, strip, and remove duplicates within the same response
                topics_list = [t.strip().lower() for t in topics_raw.split(",") if t.strip()]
                # Remove duplicates while preserving order
                seen = set()
                topics_list = [t for t in topics_list if not (t in seen or seen.add(t))]
            else:
                topics_list = []
            
            # If no topics, create one row with empty topic
            if not topics_list:
                open_long_rows.append({
                    "response_index": response_index,
                    "question_num": question_num,
                    "question_short": question_short,
                    "question_col": question_col,
                    "text": text,
                    "sentiment_raw": sentiment_raw,
                    "sentiment": sentiment if sentiment else "empty",
                    "sentiment_score": sentiment_score,
                    "topics_raw": topics_raw,
                    "topic": ""
                })
            else:
                # Create one row per topic (already deduplicated)
                for topic in topics_list:
                    open_long_rows.append({
                        "response_index": response_index,
                        "question_num": question_num,
                        "question_short": question_short,
                        "question_col": question_col,
                        "text": text,
                        "sentiment_raw": sentiment_raw,
                        "sentiment": sentiment if sentiment else "empty",
                        "sentiment_score": sentiment_score,
                        "topics_raw": topics_raw,
                        "topic": topic
                    })
    
    open_long = pd.DataFrame(open_long_rows)
    if not open_long.empty:
        # Filter out empty topics if needed (keep for now, can filter later)
        open_long = open_long[open_long["text"].notna() & (open_long["text"] != "")]
    
    # Merge ID columns to open_long and free_long
    id_cols_candidate = [
        "AgentName", "InviteeFullName",
        "Acente Açılış Tarihi", "Acente Bölge", "Acente İli",
        "enSegmenti", "Sınıf", "Grup", "Harf Skoru",
    ]
    id_cols_present = [c for c in id_cols_candidate if c in df.columns]
    
    if id_cols_present:
        df_ids = df[id_cols_present].copy()
        df_ids = df_ids.reset_index().rename(columns={"index": "response_index"})
        
        if not open_long.empty:
            open_long = open_long.merge(df_ids, on="response_index", how="left")
    
    # Create free_long master table
    free_long_rows = []
    
    for meta in open_question_meta:
        question_num = meta["num"]
        question_short = meta.get("question_short", f"Soru {question_num}")
        question_col = meta.get("question_col")
        topics_col = meta.get("topics_col")
        free_topics_col = meta.get("free_topics_col")
            
        if not free_topics_col or free_topics_col not in df.columns:
            continue
        
        # Filter rows with free_topics
        mask = df[free_topics_col].notna() & (df[free_topics_col].astype(str).str.strip() != "")
        relevant_rows = df_reset[mask].copy()
        
        if relevant_rows.empty:
            continue
        
        for idx, row in relevant_rows.iterrows():
            response_index = row.get("index", idx)
            
            # Get free topics
            free_topics_str = str(row[free_topics_col]).strip()
            if free_topics_str and free_topics_str != "nan":
                free_topics_list = [t.strip() for t in free_topics_str.split(",") if t.strip()]
            else:
                free_topics_list = []
            
            # Get fixed topics (from topics_col)
            if topics_col and topics_col in df.columns:
                topics_str = str(row[topics_col]) if pd.notna(row[topics_col]) else ""
                if topics_str and topics_str != "nan":
                    fixed_topics_list = [t.strip().lower() for t in topics_str.split(",") if t.strip()]
                else:
                    fixed_topics_list = []
            else:
                fixed_topics_list = []
            
            # Create combinations: each free_topic with each fixed_topic (or 'diger' if no fixed)
            if not free_topics_list:
                continue
            
            for free_topic in free_topics_list:
                if fixed_topics_list:
                    for fixed_topic in fixed_topics_list:
                        free_long_rows.append({
                            "response_index": response_index,
                            "question_num": question_num,
                            "question_short": question_short,
                            "question_col": question_col if question_col else "",
                            "topic": fixed_topic,
                            "free_topic": free_topic
                        })
                else:
                    # No fixed topic, use 'diger'
                    free_long_rows.append({
                        "response_index": response_index,
                        "question_num": question_num,
                        "question_short": question_short,
                        "question_col": question_col if question_col else "",
                        "topic": "diger",
                        "free_topic": free_topic
                    })
    
    free_long = pd.DataFrame(free_long_rows)
    
    # Merge ID columns to free_long
    if id_cols_present and not free_long.empty:
        free_long = free_long.merge(df_ids, on="response_index", how="left")
    
    # Load Likert & NPS data from likert+NPS sheet only (no fallback parsing)
    likert_numeric_cols = {}
    LIKERT_GROUPS_TO_USE = LIKERT_GROUPS
    likert_error = None
    
    if df_likert_scores is None:
        likert_error = "Excel dosyasında 'likert+NPS' sheet'i bulunamadı. Likert analizleri için önce bu sheet oluşturulmalıdır."
    else:
        if len(df_likert_scores) != len(df):
            likert_error = (
                f"'likert+NPS' sheet satır sayısı ile 'scored' sheet satır sayısı uyuşmuyor "
                f"(scored={len(df)}, likert+NPS={len(df_likert_scores)})."
            )
        else:
            # Attach likert+NPS columns to df by row order
            for col in df_likert_scores.columns:
                if col not in df.columns:
                    df[col] = df_likert_scores[col].values
            
            # Build likert_numeric_cols from Q<num>_num pattern
            for col in df_likert_scores.columns:
                m = re.match(r"^Q(\d+)_num$", str(col))
                if m:
                    num = int(m.group(1))
                    likert_numeric_cols[num] = col
            
            # Group averages should already be in df from the Excel file
            # If missing, try to compute them using group names from konular sheet
            # First, try to use group names from konular sheet if available
            if df_konular is not None and "Konu" in df_konular.columns:
                # Get unique group names from konular sheet
                unique_groups = df_konular["Konu"].dropna().unique()
                for group_name in unique_groups:
                    group_name_str = str(group_name).strip()
                    avg_col = f"{group_name_str}_Likert_Avg"
                    if avg_col not in df.columns:
                        # Find question numbers for this group
                        group_questions = df_konular[df_konular["Konu"] == group_name]["Soru No"].dropna()
                        question_nums = [int(q) for q in group_questions if pd.notna(q)]
                        cols = [likert_numeric_cols.get(num) for num in question_nums if num in likert_numeric_cols]
                        cols = [c for c in cols if c and c in df.columns]
                        if cols:
                            df[avg_col] = df[cols].mean(axis=1, skipna=True)
            else:
                # Fallback to old LIKERT_GROUPS if konular sheet not available
                for group_name, nums in LIKERT_GROUPS.items():
                    avg_col = f"{group_name}_Likert_Avg"
                    if avg_col not in df.columns:
                        cols = [likert_numeric_cols.get(num) for num in nums if num in likert_numeric_cols]
                        cols = [c for c in cols if c and c in df.columns]
                        if cols:
                            df[avg_col] = df[cols].mean(axis=1, skipna=True)
    
    # Build UI metadata from konular sheet
    likert_groups_ui = {}
    likert_question_texts = {}
    
    if df_konular is not None and likert_numeric_cols:
        # Determine question text column
        question_text_col = None
        if "Soru Metni" in df_konular.columns:
            question_text_col = "Soru Metni"
        elif "Soru" in df_konular.columns:
            question_text_col = "Soru"
        
        if question_text_col and "Soru No" in df_konular.columns and "Konu" in df_konular.columns:
            for idx, row in df_konular.iterrows():
                try:
                    soru_no = row["Soru No"]
                    konu = row["Konu"]
                    soru_metni = row[question_text_col]
                    
                    # Skip rows with missing values
                    if pd.isna(soru_no) or pd.isna(konu) or pd.isna(soru_metni):
                        continue
                    
                    # Convert soru_no to integer
                    try:
                        num = int(soru_no)
                    except (ValueError, TypeError):
                        continue
                    
                    # Only include questions that have numeric columns
                    if num not in likert_numeric_cols:
                        continue
                    
                    # Use exact string from Konu column as group name
                    group_name_ui = str(konu).strip()
                    
                    # Initialize list if needed
                    if group_name_ui not in likert_groups_ui:
                        likert_groups_ui[group_name_ui] = []
                    
                    # Add question number to group
                    if num not in likert_groups_ui[group_name_ui]:
                        likert_groups_ui[group_name_ui].append(num)
                    
                    # Store question text
                    likert_question_texts[num] = str(soru_metni).strip()
                except Exception:
                    continue
            
            # Deduplicate and sort question numbers for each group
            for group_name in likert_groups_ui:
                likert_groups_ui[group_name] = sorted(list(set(likert_groups_ui[group_name])))
    
    # Create thematic open sentiment columns using sentiment_score_col from open_question_meta
    # Map open-ended question numbers to their corresponding Likert group names
    # First, try to get group names from konular sheet for these specific questions
    open_to_likert_group = {}
    thematic_question_nums = ["12", "18", "25", "32"]
    
    if df_konular is not None and "Soru No" in df_konular.columns and "Konu" in df_konular.columns:
        for num_str in thematic_question_nums:
            num = int(num_str)
            # Check if this question number is in konular sheet
            matching_rows = df_konular[df_konular["Soru No"] == num]
            if not matching_rows.empty:
                group_name = str(matching_rows.iloc[0]["Konu"]).strip()
                open_to_likert_group[num_str] = group_name
            else:
                # Try to find matching group name from likert_groups_ui based on keywords
                keywords_map = {
                    "12": ["Kasko", "kasko"],
                    "18": ["TSS", "Tamamlayıcı", "Sağlık"],
                    "25": ["Konut", "konut"],
                    "32": ["Ticari", "Kurumsal", "ticari"]
                }
                keywords = keywords_map.get(num_str, [])
                for group_name_ui in likert_groups_ui.keys():
                    if any(kw in group_name_ui for kw in keywords):
                        open_to_likert_group[num_str] = group_name_ui
                        break
    
    # Create thematic sentiment columns using actual group names
    for num_str in thematic_question_nums:
        # Find the meta entry for this question number
        meta_for_num = next((m for m in open_question_meta if m["num"] == num_str), None)
        if meta_for_num is not None:
            score_col = meta_for_num.get("sentiment_score_col")
            if score_col and score_col in df.columns:
                # Use group name from mapping if available
                group_name = open_to_likert_group.get(num_str)
                if group_name:
                    # Create column with group name
                    df[f"{group_name}_OpenSentiment"] = df[score_col]
                else:
                    # Fallback: try to find matching Likert group column and use that name
                    # Look for Likert average columns that might match
                    for likert_col in df.columns:
                        if likert_col.endswith("_Likert_Avg"):
                            keywords_map = {
                                "12": ["Kasko"],
                                "18": ["TSS", "Tamamlayıcı"],
                                "25": ["Konut"],
                                "32": ["Ticari", "Kurumsal"]
                            }
                            keywords = keywords_map.get(num_str, [])
                            if any(kw in likert_col for kw in keywords):
                                group_name = likert_col.replace("_Likert_Avg", "")
                                df[f"{group_name}_OpenSentiment"] = df[score_col]
                                break
                    else:
                        # Final fallback to old names
                        old_names = {"12": "Kasko", "18": "TSS", "25": "Konut", "32": "Ticari"}
                        old_name = old_names.get(num_str)
                        if old_name:
                            df[f"{old_name}_OpenSentiment"] = df[score_col]
    
    # Store metadata and master tables
    df.attrs["open_question_meta"] = open_question_meta
    df.attrs["likert_groups"] = LIKERT_GROUPS_TO_USE if likert_error is None else {}
    df.attrs["likert_numeric_cols"] = likert_numeric_cols
    df.attrs["likert_groups_ui"] = likert_groups_ui if likert_groups_ui else {}
    df.attrs["likert_question_texts"] = likert_question_texts
    df.attrs["likert_error"] = likert_error
    df.attrs["open_long"] = open_long
    df.attrs["free_long"] = free_long
    
    return df, summary_df if 'summary_df' in locals() else pd.DataFrame()


# ============================================================================
# DATA PREPARATION FUNCTIONS (Removed - now using open_long and free_long)
# ============================================================================


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">📊 Anket Sentiment Analizi Dashboard</h1>', unsafe_allow_html=True)
    
    # Veri yükleme
    df, summary_df = load_data()
    
    if df is None:
        st.error("Veri yüklenemedi. Lütfen dosya yolunu kontrol edin.")
        return
    
    # Genel bilgi kutusu
    with st.expander("ℹ️ Dashboard Hakkında", expanded=False):
        st.markdown("""
        **Anket Sentiment Analizi Dashboard**
        
        Bu dashboard, anket yanıtlarının sentiment (duygu durumu) ve topic (konu) analizlerini görselleştirir.
        
        **Kullanım:**
        - Sol taraftaki filtrelerden istediğiniz soruları seçerek analizleri daraltabilirsiniz
        - Her sekmenin başındaki "📖 Terim Açıklamaları" bölümünden kullanılan terimlerin açıklamalarını görebilirsiniz
        - Grafiklerin üzerine gelerek detaylı bilgileri görebilirsiniz
        - Grafiklerin sağ üst köşesindeki araçları kullanarak zoom, pan, indirme gibi işlemler yapabilirsiniz
        """)
    
    # Sidebar - Filtreler
    st.sidebar.header("🔍 Filtreler ve Seçenekler")
    
    # Soru seçimi
    open_question_meta = df.attrs.get("open_question_meta", [])
    question_options = [meta.get("question_short", f"Soru {meta['num']}") for meta in open_question_meta]
    
    if not question_options:
        st.warning("Açık uçlu soru bulunamadı. Veri yapısını kontrol edin.")
        return
    
    # Initialize session state for question selection
    if 'selected_questions' not in st.session_state:
        st.session_state.selected_questions = question_options.copy()
    
    # Butonlar
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("✅ Tümünü Seç", use_container_width=True):
            st.session_state.selected_questions = question_options.copy()
            st.rerun()
    with col2:
        if st.button("🗑️ Seçimleri Temizle", use_container_width=True):
            st.session_state.selected_questions = []
            st.rerun()
    
    selected_questions = st.sidebar.multiselect(
        "Soruları Seçin",
        options=question_options,
        default=st.session_state.selected_questions,
        help="Analizleri belirli sorularla sınırlamak için soruları seçin. Hiçbir şey seçmezseniz tüm sorular analiz edilir."
    )
    
    # Update session state when selection changes
    if selected_questions != st.session_state.selected_questions:
        st.session_state.selected_questions = selected_questions
    
    # Acenta Filtreleri
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏢 Acenta Filtreleri")
    
    # Bölge filtresi
    selected_bolgeler = []
    if "Acente Bölge" in df.columns:
        bolgeler = sorted(df["Acente Bölge"].dropna().unique())
        if 'selected_bolgeler_state' not in st.session_state:
            st.session_state.selected_bolgeler_state = bolgeler.copy()
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("✅ Tümünü Seç", key="bolge_all", use_container_width=True):
                st.session_state.selected_bolgeler_state = bolgeler.copy()
                st.rerun()
        with col2:
            if st.button("🗑️ Temizle", key="bolge_clear", use_container_width=True):
                st.session_state.selected_bolgeler_state = []
                st.rerun()
        
        selected_bolgeler = st.sidebar.multiselect(
            "Bölge", 
            bolgeler, 
            default=st.session_state.selected_bolgeler_state,
            key="multiselect_bolge"
        )
        st.session_state.selected_bolgeler_state = selected_bolgeler
    
    # İl filtresi
    selected_iller = []
    if "Acente İli" in df.columns:
        iller = sorted(df["Acente İli"].dropna().unique())
        if 'selected_iller_state' not in st.session_state:
            st.session_state.selected_iller_state = iller.copy()
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("✅ Tümünü Seç", key="il_all", use_container_width=True):
                st.session_state.selected_iller_state = iller.copy()
                st.rerun()
        with col2:
            if st.button("🗑️ Temizle", key="il_clear", use_container_width=True):
                st.session_state.selected_iller_state = []
                st.rerun()
        
        selected_iller = st.sidebar.multiselect(
            "İl", 
            iller, 
            default=st.session_state.selected_iller_state,
            key="multiselect_il"
        )
        st.session_state.selected_iller_state = selected_iller
    
    # Segment filtresi
    selected_segmentler = []
    if "enSegmenti" in df.columns:
        segmentler = sorted(df["enSegmenti"].dropna().unique())
        if 'selected_segmentler_state' not in st.session_state:
            st.session_state.selected_segmentler_state = segmentler.copy()
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("✅ Tümünü Seç", key="segment_all", use_container_width=True):
                st.session_state.selected_segmentler_state = segmentler.copy()
                st.rerun()
        with col2:
            if st.button("🗑️ Temizle", key="segment_clear", use_container_width=True):
                st.session_state.selected_segmentler_state = []
                st.rerun()
        
        selected_segmentler = st.sidebar.multiselect(
            "Segment", 
            segmentler, 
            default=st.session_state.selected_segmentler_state,
            key="multiselect_segment"
        )
        st.session_state.selected_segmentler_state = selected_segmentler
    
    # Sınıf filtresi
    selected_siniflar = []
    if "Sınıf" in df.columns:
        siniflar = sorted(df["Sınıf"].dropna().unique())
        if 'selected_siniflar_state' not in st.session_state:
            st.session_state.selected_siniflar_state = siniflar.copy()
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("✅ Tümünü Seç", key="sinif_all", use_container_width=True):
                st.session_state.selected_siniflar_state = siniflar.copy()
                st.rerun()
        with col2:
            if st.button("🗑️ Temizle", key="sinif_clear", use_container_width=True):
                st.session_state.selected_siniflar_state = []
                st.rerun()
        
        selected_siniflar = st.sidebar.multiselect(
            "Sınıf", 
            siniflar, 
            default=st.session_state.selected_siniflar_state,
            key="multiselect_sinif"
        )
        st.session_state.selected_siniflar_state = selected_siniflar
    
    # Grup filtresi
    selected_gruplar = []
    if "Grup" in df.columns:
        gruplar = sorted(df["Grup"].dropna().unique())
        if 'selected_gruplar_state' not in st.session_state:
            st.session_state.selected_gruplar_state = gruplar.copy()
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("✅ Tümünü Seç", key="grup_all", use_container_width=True):
                st.session_state.selected_gruplar_state = gruplar.copy()
                st.rerun()
        with col2:
            if st.button("🗑️ Temizle", key="grup_clear", use_container_width=True):
                st.session_state.selected_gruplar_state = []
                st.rerun()
        
        selected_gruplar = st.sidebar.multiselect(
            "Grup", 
            gruplar, 
            default=st.session_state.selected_gruplar_state,
            key="multiselect_grup"
        )
        st.session_state.selected_gruplar_state = selected_gruplar
    
    # Harf Skoru filtresi
    selected_harfler = []
    if "Harf Skoru" in df.columns:
        harfler = sorted(df["Harf Skoru"].dropna().unique())
        if 'selected_harfler_state' not in st.session_state:
            st.session_state.selected_harfler_state = harfler.copy()
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("✅ Tümünü Seç", key="harf_all", use_container_width=True):
                st.session_state.selected_harfler_state = harfler.copy()
                st.rerun()
        with col2:
            if st.button("🗑️ Temizle", key="harf_clear", use_container_width=True):
                st.session_state.selected_harfler_state = []
                st.rerun()
        
        selected_harfler = st.sidebar.multiselect(
            "Harf Skoru", 
            harfler, 
            default=st.session_state.selected_harfler_state,
            key="multiselect_harf"
        )
        st.session_state.selected_harfler_state = selected_harfler
    
    selected_date_range = None
    if "Acente Açılış Tarihi" in df.columns:
        tarihler = pd.to_datetime(df["Acente Açılış Tarihi"], errors="coerce")
        min_t, max_t = tarihler.min(), tarihler.max()
        if pd.notna(min_t) and pd.notna(max_t):
            selected_date_range = st.sidebar.date_input(
                "Acente açılış tarihi aralığı",
                (min_t.date(), max_t.date())
            )
    
    # Get master tables
    open_long = df.attrs.get("open_long", pd.DataFrame())
    free_long = df.attrs.get("free_long", pd.DataFrame())
    
    # Apply global filters
    mask = pd.Series(True, index=df.index)
    
    if "Acente Bölge" in df.columns and selected_bolgeler:
        mask &= df["Acente Bölge"].isin(selected_bolgeler)
    
    if "Acente İli" in df.columns and selected_iller:
        mask &= df["Acente İli"].isin(selected_iller)
    
    if "enSegmenti" in df.columns and selected_segmentler:
        mask &= df["enSegmenti"].isin(selected_segmentler)
    
    if "Sınıf" in df.columns and selected_siniflar:
        mask &= df["Sınıf"].isin(selected_siniflar)
    
    if "Grup" in df.columns and selected_gruplar:
        mask &= df["Grup"].isin(selected_gruplar)
    
    if "Harf Skoru" in df.columns and selected_harfler:
        mask &= df["Harf Skoru"].isin(selected_harfler)
    
    if selected_date_range is not None and isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
        start_date, end_date = selected_date_range
        tarihler = pd.to_datetime(df["Acente Açılış Tarihi"], errors="coerce")
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        mask &= (tarihler >= start_ts) & (tarihler <= end_ts)
    
    df_filtered = df[mask].copy()
    
    if not open_long.empty and "response_index" in open_long.columns:
        open_long = open_long[open_long["response_index"].isin(df_filtered.index)].copy()
    
    if not free_long.empty and "response_index" in free_long.columns:
        free_long = free_long[free_long["response_index"].isin(df_filtered.index)].copy()
    
    # Preserve attrs and update with filtered long tables
    df_filtered.attrs.update(df.attrs)
    df_filtered.attrs["open_long"] = open_long
    df_filtered.attrs["free_long"] = free_long
    
    df = df_filtered
    
    # Tab seçimi - 9 consolidated tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Genel Bakış",
        "Açık Uç – Sentiment",
        "Topic Analizi",
        "Free Topics & Taksonomi",
        "Temsilci Verbatim",
        "Likert Analizleri",
        "Likert vs Açık Uç",
        "Wordcloud",
        "Ham Veri"
    ])
    
    # TAB 1: Genel Bakış
    with tab1:
        st.header("📈 Genel Bakış")
        
        # Terim Açıklamaları
        with st.expander("📖 Terim Açıklamaları", expanded=False):
            st.markdown("""
            **Genel Bakış Terimleri:**
            - **Toplam Katılımcı Sayısı**: Ankete katılan toplam kişi sayısı
            - **Açık Uçlu Sorulara Verilen Toplam Yanıt**: Açık uçlu sorulara verilen toplam yanıt sayısı
            - **Pozitif/Negatif**: Sentiment analizi sonucuna göre pozitif/negatif yanıt sayısı
            - **Topic**: Açık uçlu yanıtlardan yakalanan taksonomi kelimeleri
            """)
        
        # Ana metrikler
        col1, col2, col3, col4 = st.columns(4)
        
        total_responses = len(df)
        
        if not open_long.empty:
            # Açık uçlu sorulara verilen toplam yanıt sayısı (unique kişi değil, toplam yanıt)
            total_with_open = len(open_long)
            # Filter out empty sentiment for sentiment counts
            valid_sentiment = open_long[open_long["sentiment"] != "empty"]
            sentiment_counts = valid_sentiment["sentiment"].value_counts()
            total_positive = sentiment_counts.get("positive", 0)
            total_negative = sentiment_counts.get("negative", 0)
            total_neutral = sentiment_counts.get("neutral", 0)
            total_mixed = sentiment_counts.get("mixed", 0)
            total_sentiment = total_positive + total_negative + total_neutral + total_mixed
        else:
            total_with_open = 0
            total_positive = total_negative = total_neutral = total_mixed = 0
            total_sentiment = 0
        
        with col1:
            st.metric("Toplam Katılımcı Sayısı", total_responses)
        with col2:
            st.metric("Açık Uçlu Sorulara Verilen Toplam Yanıt", total_with_open)
        with col3:
            st.metric("Pozitif", total_positive)
        with col4:
            st.metric("Negatif", total_negative)
        
        # Özet grafikler
        if not open_long.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Global sentiment dağılımı
                valid_sentiment = open_long[open_long["sentiment"] != "empty"]
                if not valid_sentiment.empty:
                    sentiment_totals = valid_sentiment["sentiment"].value_counts().reset_index()
                    sentiment_totals.columns = ["sentiment", "count"]
                    
                    fig_pie = px.pie(
                        sentiment_totals,
                        values='count',
                        names='sentiment',
                        title='Açık Uçlu Sorularda Genel Sentiment Dağılımı',
                        color='sentiment',
                        color_discrete_map=SENTIMENT_COLORS
                    )
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # En sık geçen topic (Top 20)
                if not open_long.empty and "topic" in open_long.columns:
                    topic_counts_all = open_long[open_long["topic"] != ""]["topic"].value_counts()
                    if not topic_counts_all.empty:
                        # Limit to Top 20
                        max_n_genel = min(20, len(topic_counts_all))
                        topic_counts_top = topic_counts_all.head(max_n_genel)
                        
                        top_topics_df = pd.DataFrame({
                            'topic': topic_counts_top.index,
                            'count': topic_counts_top.values
                        })
                        
                        fig_top_topics = px.bar(
                            top_topics_df,
                            x='count',
                            y='topic',
                            orientation='h',
                            title=f'En Sık Geçen Topic\'ler (Top {max_n_genel})',
                            labels={'count': 'Frekans', 'topic': 'Topic'}
                        )
                        # Calculate height based on number of topics (min 400, ~40px per topic)
                        chart_height = max(400, len(top_topics_df) * 40)
                        fig_top_topics.update_layout(height=chart_height, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_top_topics, use_container_width=True)
    
    # TAB 2: Açık Uç – Sentiment
    with tab2:
        st.header("Açık Uç – Sentiment Analizi")
        
        # Terim Açıklamaları
        with st.expander("📖 Terim Açıklamaları", expanded=False):
            st.markdown("""
            **Sentiment Analizi Terimleri:**
            - **Sentiment (Duygu Durumu)**: Metinlerin pozitif, negatif, nötr veya karma duygu durumuna göre sınıflandırılması
            - **Sentiment Skoru**: -1.0 (çok negatif) ile +1.0 (çok pozitif) arasında değişen sayısal değer
            - **Positive**: Olumlu duygu ifade eden yanıtlar
            - **Negative**: Olumsuz duygu ifade eden yanıtlar
            - **Neutral**: Duygu ifade etmeyen veya tarafsız yanıtlar
            - **Mixed**: Hem olumlu hem olumsuz duygu içeren yanıtlar
            """)
        
        if open_long.empty:
            st.info("Açık uçlu veri bulunamadı.")
        else:
            # Filter by selected questions
            filtered_open = open_long.copy()
            if selected_questions:
                filtered_open = filtered_open[filtered_open["question_short"].isin(selected_questions)]
            
            # Remove empty sentiment
            filtered_open = filtered_open[filtered_open["sentiment"] != "empty"]
            
            if filtered_open.empty:
                st.info("Seçilen sorular için veri bulunamadı.")
            else:
                # Soru bazında sentiment dağılımı
                st.subheader("Soru Bazında Sentiment Dağılımı")
                sentiment_by_question = filtered_open.groupby(["question_short", "sentiment"]).size().reset_index(name="count")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Stacked bar chart
                    fig_stacked = px.bar(
                        sentiment_by_question,
                        x='question_short',
                        y='count',
                        color='sentiment',
                        title='Soru Bazında Sentiment Dağılımı',
                        labels={'question_short': 'Soru', 'count': 'Yanıt Sayısı', 'sentiment': 'Sentiment'},
                        color_discrete_map=SENTIMENT_COLORS
                    )
                    fig_stacked.update_layout(height=500, xaxis_tickangle=-45)
                    st.plotly_chart(fig_stacked, use_container_width=True)
                
                with col2:
                    # Heatmap
                    pivot_sentiment = sentiment_by_question.pivot_table(
                        index='question_short',
                        columns='sentiment',
                        values='count',
                        fill_value=0
                    )
                    
                    if not pivot_sentiment.empty:
                        fig_heatmap = px.imshow(
                            pivot_sentiment,
                            labels=dict(x="Sentiment", y="Soru", color="Yanıt Sayısı"),
                            title="Sentiment Heatmap (Soru x Sentiment)",
                            color_continuous_scale="RdYlGn",
                            aspect="auto"
                        )
                        fig_heatmap.update_layout(height=500)
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Sentiment skorları
                st.subheader("Sentiment Skorları")
                
                # Filter valid scores
                score_data = filtered_open[filtered_open["sentiment_score"].notna()].copy()
                
                if not score_data.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Histogram
                        fig_hist = px.histogram(
                            score_data,
                            x='sentiment_score',
                            nbins=30,
                            title='Sentiment Skorları Dağılımı',
                            labels={'sentiment_score': 'Sentiment Skoru (-1.0 ile +1.0 arası)', 'count': 'Frekans'},
                            color_discrete_sequence=['#3498db']
                        )
                        fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Nötr")
                        fig_hist.update_layout(height=400)
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        # Box plot by question
                        fig_box = px.box(
                            score_data,
                            x='question_short',
                            y='sentiment_score',
                            title='Soru Bazında Sentiment Skoru Dağılımı',
                            labels={'question_short': 'Soru', 'sentiment_score': 'Sentiment Skoru'}
                        )
                        fig_box.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Nötr")
                        fig_box.update_layout(height=400, xaxis_tickangle=-45)
                        st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Ortalama skorlar
                    avg_scores = score_data.groupby('question_short')['sentiment_score'].mean().sort_values()
                    if not avg_scores.empty:
                        avg_scores_df = pd.DataFrame({
                            'question': avg_scores.index,
                            'score': avg_scores.values
                        })
                        
                        fig_avg = px.bar(
                            avg_scores_df,
                            x='score',
                            y='question',
                            orientation='h',
                            title='Soru Bazında Ortalama Sentiment Skorları',
                            labels={'score': 'Ortalama Sentiment Skoru', 'question': 'Soru'},
                            color='score',
                            color_continuous_scale="RdYlGn"
                        )
                        fig_avg.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Nötr")
                        fig_avg.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_avg, use_container_width=True)
    
    # TAB 3: Topic Analizi
    with tab3:
        st.header("Topic Analizi")
        
        # Terim Açıklamaları
        with st.expander("📖 Terim Açıklamaları", expanded=False):
            st.markdown("""
            **Topic Analizi Terimleri:**
            - **Topic**: Açık uçlu yanıtlardan otomatik olarak çıkarılan konu kategorileri
            - **Topic Frekansı**: Bir topic'in kaç kez geçtiğinin sayısı
            - **Pain/Gain Analizi**: Topic'lerin pozitif/negatif yanıt oranlarına göre analizi
            - **Mention**: Bir topic'in bir yanıtta geçmesi
            - **Negatif Analiz**: Sadece negatif sentiment içeren yanıtlardaki topic'lerin analizi
            """)
        
        if open_long.empty:
            st.info("Topic verisi bulunamadı.")
        else:
            # Filter by selected questions
            filtered_open = open_long.copy()
            if selected_questions:
                filtered_open = filtered_open[filtered_open["question_short"].isin(selected_questions)]
            
            # Filter out empty topics
            filtered_topics = filtered_open[filtered_open["topic"] != ""].copy()
            
            if filtered_topics.empty:
                st.info("Seçilen sorular için topic verisi bulunamadı.")
            else:
                # Topic frekansları (global)
                st.subheader("Topic Frekansları")
                topic_counts_all = filtered_topics.groupby("topic").size().sort_values(ascending=False)
                if not topic_counts_all.empty:
                    max_n = min(20, len(topic_counts_all))
                    top_n = max_n
                    topic_counts = topic_counts_all.head(top_n)
                    
                    if not topic_counts.empty:
                        top_topics_df = pd.DataFrame({
                            'topic': topic_counts.index,
                            'count': topic_counts.values
                        })
                        
                    fig_top_topics = px.bar(
                        top_topics_df,
                        x='count',
                        y='topic',
                        orientation='h',
                        title=f'En Sık Geçen Topic\'ler (Top {top_n})',
                        labels={'count': 'Frekans', 'topic': 'Topic'}
                    )
                    fig_top_topics.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_top_topics, use_container_width=True)
                
                # Soru x Topic Matrisi
                st.subheader("Soru x Topic Matrisi")
                question_topic_counts = filtered_topics.groupby(["question_short", "topic"]).size().reset_index(name="count")
                pivot_table = question_topic_counts.pivot(index="topic", columns="question_short", values="count").fillna(0)
                
                # Show top topics (user-configurable)
                if not topic_counts_all.empty:
                    max_n_heatmap = min(20, len(topic_counts_all))
                    top_n_heatmap = max_n_heatmap
                    top_topics_heatmap = topic_counts_all.head(top_n_heatmap).index
                    pivot_filtered = pivot_table[pivot_table.index.isin(top_topics_heatmap)]
                    
                    if not pivot_filtered.empty:
                        fig_heatmap = px.imshow(
                            pivot_filtered,
                            labels=dict(x="Soru", y="Topic", color="Mention Sayısı"),
                            title=f"Soru x Topic Heatmap (Top {top_n_heatmap})",
                            color_continuous_scale="Viridis",
                            aspect="auto"
                        )
                        fig_heatmap.update_layout(height=600)
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Topic Bazlı Pain/Gain
                st.subheader("Topic Bazlı Pain/Gain Analizi")
                
                # Calculate topic summary
                topic_summary = []
                for topic in filtered_topics["topic"].unique():
                    topic_data = filtered_topics[filtered_topics["topic"] == topic]
                    
                    total_mentions = len(topic_data)
                    sentiment_counts = topic_data["sentiment"].value_counts()
                    positive_count = sentiment_counts.get("positive", 0)
                    negative_count = sentiment_counts.get("negative", 0)
                    neutral_count = sentiment_counts.get("neutral", 0)
                    mixed_count = sentiment_counts.get("mixed", 0)
                    
                    avg_sentiment_score = topic_data["sentiment_score"].mean() if topic_data["sentiment_score"].notna().any() else np.nan
                    positive_ratio = positive_count / total_mentions if total_mentions > 0 else 0
                    negative_ratio = negative_count / total_mentions if total_mentions > 0 else 0
                    
                    topic_summary.append({
                        "topic": topic,
                        "total_mentions": total_mentions,
                        "positive_count": positive_count,
                        "negative_count": negative_count,
                        "neutral_count": neutral_count,
                        "mixed_count": mixed_count,
                        "positive_ratio": positive_ratio,
                        "negative_ratio": negative_ratio,
                        "avg_sentiment_score": avg_sentiment_score
                    })
                
                if topic_summary:
                    summary_df_pain = pd.DataFrame(topic_summary)
                    summary_df_pain = summary_df_pain.sort_values("avg_sentiment_score", ascending=True)
                    summary_df_pain = summary_df_pain.reset_index(drop=True)
                    summary_df_pain.index.name = "Sıra"
                    summary_df_pain = summary_df_pain.reset_index()
                    summary_df_pain["Sıra"] = summary_df_pain["Sıra"] + 1
                    
                    # Limit to Top 20 for display
                    max_n_pain = min(20, len(summary_df_pain))
                    summary_df_pain_top = summary_df_pain.head(max_n_pain)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Average sentiment score
                        fig_pain = px.bar(
                            summary_df_pain_top,
                            x='avg_sentiment_score',
                            y='topic',
                            orientation='h',
                            title=f'Topic Bazlı Ortalama Sentiment Skoru (Top {max_n_pain})',
                            labels={'avg_sentiment_score': 'Ortalama Sentiment Skoru', 'topic': 'Topic'},
                            color='avg_sentiment_score',
                            color_continuous_scale="RdYlGn"
                        )
                        fig_pain.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Nötr")
                        fig_pain.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_pain, use_container_width=True)
                    
                    with col2:
                        # Reorder columns to show Sıra first
                        display_cols = ["Sıra", "topic", "total_mentions", "positive_count", "negative_count", 
                                       "neutral_count", "mixed_count", "positive_ratio", "negative_ratio", "avg_sentiment_score"]
                        st.dataframe(summary_df_pain_top[display_cols], use_container_width=True, height=600, hide_index=True)
                
                # Negatif-only view (checkbox toggle)
                st.subheader("Negatif Analiz")
                show_negative_only = st.checkbox("Sadece negatif mention'ları göster", value=False)
                
                if show_negative_only:
                    negative_topics = filtered_topics[filtered_topics["sentiment"] == "negative"].copy()
                    
                    if not negative_topics.empty:
                        neg_topic_counts_all = negative_topics.groupby("topic").size().sort_values(ascending=False)
                        if not neg_topic_counts_all.empty:
                            max_n_neg = min(20, len(neg_topic_counts_all))
                            top_n_neg = max_n_neg
                            neg_topic_counts = neg_topic_counts_all.head(top_n_neg)
                            
                            if not neg_topic_counts.empty:
                                neg_topics_df = pd.DataFrame({
                                    'topic': neg_topic_counts.index,
                                    'count': neg_topic_counts.values
                                })
                                
                                fig_neg_topics = px.bar(
                                    neg_topics_df,
                                    x='count',
                                    y='topic',
                                    orientation='h',
                                    title=f'Negatif Yanıtlarda En Sık Geçen Topic\'ler (Top {top_n_neg})',
                                    labels={'count': 'Frekans', 'topic': 'Topic'},
                                    color='count',
                                    color_continuous_scale="Reds"
                                )
                                fig_neg_topics.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                                st.plotly_chart(fig_neg_topics, use_container_width=True)
                                
                                # Negatif heatmap
                                neg_question_topic = negative_topics.groupby(["question_short", "topic"]).size().reset_index(name="count")
                                neg_pivot = neg_question_topic.pivot(index="topic", columns="question_short", values="count").fillna(0)
                                
                                max_n_neg_heatmap = min(20, len(neg_topic_counts_all))
                                top_n_neg_heatmap = max_n_neg_heatmap
                                top_neg_topics = neg_topic_counts_all.head(top_n_neg_heatmap).index
                                neg_pivot_filtered = neg_pivot[neg_pivot.index.isin(top_neg_topics)]
                                
                                if not neg_pivot_filtered.empty:
                                    fig_neg_heatmap = px.imshow(
                                        neg_pivot_filtered,
                                        labels=dict(x="Soru", y="Topic", color="Frekans"),
                                        title=f"Negatif Topic Heatmap (Soru x Topic) - Top {top_n_neg_heatmap}",
                                        color_continuous_scale="Reds",
                                        aspect="auto"
                                    )
                                    fig_neg_heatmap.update_layout(height=500)
                                    st.plotly_chart(fig_neg_heatmap, use_container_width=True)
                                else:
                                    st.info("Negatif topic verisi bulunamadı.")
                        else:
                            st.info("Negatif topic verisi bulunamadı.")
    
    # TAB 4: Free Topics & Taksonomi
    with tab4:
        st.header("Free Topics & Taksonomi Analizi")
        
        # Terim Açıklamaları
        with st.expander("📖 Terim Açıklamaları", expanded=False):
            st.markdown("""
            **Free Topics & Taksonomi Terimleri:**
            - **Free Topics**: Açık uçlu yanıtlardan gpt tarafından çıkarılan serbest formdaki konu ifadeleri
            - **Fixed Topics (Taksonomi)**: Önceden tanımlanmış konu kategorileri
            - **Co-occurrence**: Free topic ve fixed topic'lerin aynı yanıtta birlikte geçmesi
            - **Heatmap**: İki değişken arasındaki ilişkiyi görselleştiren ısı haritası
            """)
        
        if free_long.empty:
            st.info("Free topics verisi bulunamadı.")
        else:
            # Filter by selected questions
            filtered_free = free_long.copy()
            if selected_questions:
                filtered_free = filtered_free[filtered_free["question_short"].isin(selected_questions)]
            
            if filtered_free.empty:
                st.info("Seçilen sorular için free topics verisi bulunamadı.")
            else:
                # Free topics frekansları
                st.subheader("Free Topics Frekansları")
                free_topic_counts_all = filtered_free.groupby("free_topic").size().sort_values(ascending=False)
                if not free_topic_counts_all.empty:
                    max_n_free = min(20, len(free_topic_counts_all))
                    top_n_free = max_n_free
                    free_topic_counts = free_topic_counts_all.head(top_n_free)
                    
                    if not free_topic_counts.empty:
                        top_free_df = pd.DataFrame({
                            'free_topic': free_topic_counts.index,
                            'count': free_topic_counts.values
                        })
                        
                        fig_free = px.bar(
                            top_free_df,
                            x='count',
                            y='free_topic',
                            orientation='h',
                            title=f'En Sık Geçen Free Topics (Top {top_n_free})',
                            labels={'count': 'Frekans', 'free_topic': 'Free Topic'},
                            color='count',
                            color_continuous_scale="Viridis"
                        )
                        fig_free.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_free, use_container_width=True)
            
                # Topic x Free Topic co-occurrence
                st.subheader("Topic x Free Topic İlişkisi")
                tf_counts = filtered_free.groupby(['free_topic', 'topic']).size().reset_index(name='count')
                # User-configurable Top N
                max_n_tf = min(20, len(tf_counts)) if len(tf_counts) > 0 else 0
                top_n_tf = max_n_tf
                top_tf = tf_counts.nlargest(top_n_tf, 'count') if top_n_tf > 0 else pd.DataFrame()
                
                if not top_tf.empty:
                    free_totals = filtered_free.groupby('free_topic').size()
                    top_tf['free_total'] = top_tf['free_topic'].map(free_totals)
                    top_tf['percentage'] = (top_tf['count'] / top_tf['free_total'] * 100).round(1)
                    
                    fig_tf = px.bar(
                        top_tf,
                        x='count',
                        y='free_topic',
                        color='topic',
                        orientation='h',
                        title=f'Free Topics - Fixed Topics İlişkisi (Top {top_n_tf})',
                        labels={'count': 'Frekans', 'free_topic': 'Free Topic', 'topic': 'Fixed Topic'},
                        hover_data=['percentage']
                    )
                    fig_tf.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_tf, use_container_width=True)
            
                # Heatmap
                st.subheader("Free Topics x Fixed Topics Heatmap")
                free_counts_all = filtered_free['free_topic'].value_counts()
                fixed_counts_all = filtered_free['topic'].value_counts()
                
                if not free_counts_all.empty and not fixed_counts_all.empty:
                    max_n_free_heatmap = min(20, len(free_counts_all))
                    max_n_fixed_heatmap = min(20, len(fixed_counts_all))
                    
                    top_n_free_heatmap = max_n_free_heatmap
                    top_n_fixed_heatmap = max_n_fixed_heatmap
                    
                    top_n_free_list = free_counts_all.head(top_n_free_heatmap).index
                    top_n_fixed_list = fixed_counts_all.head(top_n_fixed_heatmap).index
                    
                    tf_filtered = filtered_free[
                        (filtered_free['free_topic'].isin(top_n_free_list)) & 
                        (filtered_free['topic'].isin(top_n_fixed_list))
                    ]
                    
                    if not tf_filtered.empty:
                        tf_pivot = tf_filtered.groupby(['free_topic', 'topic']).size().reset_index(name='count')
                        tf_pivot_wide = tf_pivot.pivot(index='free_topic', columns='topic', values='count').fillna(0)
                        
                        if not tf_pivot_wide.empty:
                            fig_tf_heatmap = px.imshow(
                                tf_pivot_wide,
                                labels=dict(x="Fixed Topic (Taksonomi)", y="Free Topic (OpenAI)", color="Frekans"),
                                title=f"Free Topics x Fixed Topics İlişkisi (Top {top_n_free_heatmap} x Top {top_n_fixed_heatmap})",
                                color_continuous_scale="Plasma",
                                aspect="auto"
                            )
                            fig_tf_heatmap.update_layout(height=500)
                            st.plotly_chart(fig_tf_heatmap, use_container_width=True)
    
                # Negatif-only view for Free Topics (checkbox toggle)
                st.subheader("Free Topics - Negatif Analiz")
                show_negative_free = st.checkbox("Sadece negatif mention'ları göster (Free Topics)", value=False, key="negative_free")
                
                if show_negative_free:
                    # Get sentiment info from open_long for the same response_index
                    if not open_long.empty:
                        # Merge free_long with open_long to get sentiment
                        free_with_sentiment = filtered_free.merge(
                            open_long[["response_index", "question_num", "sentiment"]].drop_duplicates(),
                            on=["response_index", "question_num"],
                            how="left"
                        )
                        negative_free_topics = free_with_sentiment[free_with_sentiment["sentiment"] == "negative"].copy()
                        
                        if not negative_free_topics.empty:
                            neg_free_topic_counts_all = negative_free_topics.groupby("free_topic").size().sort_values(ascending=False)
                            if not neg_free_topic_counts_all.empty:
                                max_n_neg_free = min(20, len(neg_free_topic_counts_all))
                                top_n_neg_free = max_n_neg_free
                                neg_free_topic_counts = neg_free_topic_counts_all.head(top_n_neg_free)
                                
                                if not neg_free_topic_counts.empty:
                                    neg_free_topics_df = pd.DataFrame({
                                        'free_topic': neg_free_topic_counts.index,
                                        'count': neg_free_topic_counts.values
                                    })
                                    
                                    fig_neg_free = px.bar(
                                        neg_free_topics_df,
                                        x='count',
                                        y='free_topic',
                                        orientation='h',
                                        title=f'Negatif Yanıtlarda En Sık Geçen Free Topics (Top {top_n_neg_free})',
                                        labels={'count': 'Frekans', 'free_topic': 'Free Topic'},
                                        color='count',
                                        color_continuous_scale="Reds"
                                    )
                                    fig_neg_free.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                                    st.plotly_chart(fig_neg_free, use_container_width=True)
                                    
                                    # Negatif free topics heatmap
                                    neg_free_question_topic = negative_free_topics.groupby(["question_short", "free_topic"]).size().reset_index(name="count")
                                    neg_free_pivot = neg_free_question_topic.pivot(index="free_topic", columns="question_short", values="count").fillna(0)
                                    
                                    max_n_neg_free_heatmap = min(20, len(neg_free_topic_counts_all))
                                    top_n_neg_free_heatmap = max_n_neg_free_heatmap
                                    top_neg_free_topics = neg_free_topic_counts_all.head(top_n_neg_free_heatmap).index
                                    neg_free_pivot_filtered = neg_free_pivot[neg_free_pivot.index.isin(top_neg_free_topics)]
                                    
                                    if not neg_free_pivot_filtered.empty:
                                        fig_neg_free_heatmap = px.imshow(
                                            neg_free_pivot_filtered,
                                            labels=dict(x="Soru", y="Free Topic", color="Frekans"),
                                            title=f"Negatif Free Topics Heatmap (Soru x Free Topic) - Top {top_n_neg_free_heatmap}",
                                            color_continuous_scale="Reds",
                                            aspect="auto"
                                        )
                                        fig_neg_free_heatmap.update_layout(height=500)
                                        st.plotly_chart(fig_neg_free_heatmap, use_container_width=True)
                                    else:
                                        st.info("Negatif free topic verisi bulunamadı.")
                                else:
                                    st.info("Negatif free topic verisi bulunamadı.")
                            else:
                                st.info("Negatif free topic verisi bulunamadı.")
                        else:
                            st.info("Negatif free topic verisi bulunamadı.")
                    else:
                        st.info("Sentiment verisi bulunamadı.")
    
    # TAB 5: Temsilci Verbatim
    with tab5:
        st.header("💬 Temsilci Verbatim Örnekleri")
        
        # Terim Açıklamaları
        with st.expander("📖 Terim Açıklamaları", expanded=False):
            st.markdown("""
            **Verbatim Terimleri:**
            - **Verbatim**: Katılımcıların verdiği açık uçlu yanıtların aynen gösterilmesi
            """)
        
        if open_long.empty:
            st.warning("Açık uçlu veri bulunamadı.")
        else:
            # Soru seçimi
            question_options = open_long["question_short"].unique()
            selected_question_label = st.selectbox(
                "Soru Seçin",
                options=sorted(question_options)
            )
            
            # Filter by question
            question_data = open_long[open_long["question_short"] == selected_question_label].copy()
            
            # Topic filtresi (opsiyonel)
            if not question_data.empty and question_data["topic"].notna().any():
                unique_topics = sorted(question_data[question_data["topic"] != ""]["topic"].unique())
                if unique_topics:
                    selected_topics = st.multiselect(
                        "Topic Filtresi (Opsiyonel)",
                        options=unique_topics,
                        default=[]
                    )
                    
                    if selected_topics:
                        question_data = question_data[question_data["topic"].isin(selected_topics)]
            
            # Get unique responses (by response_index)
            unique_responses = question_data.drop_duplicates(subset=["response_index"]).copy()
            
            if unique_responses.empty:
                st.info("Seçilen kriterlere uygun veri bulunamadı.")
            else:
                # Sayfalama ayarları
                num_per_page = st.number_input("Sayfa başına gösterilecek sonuç sayısı", min_value=1, max_value=50, value=5, step=1)
                
                # Pozitif örnekler (önce göster)
                st.subheader("✅ Pozitif Örnekler")
                positive_responses = unique_responses[unique_responses["sentiment"] == "positive"].copy()
                
                if not positive_responses.empty:
                    total_positive = len(positive_responses)
                    num_pages_positive = (total_positive + num_per_page - 1) // num_per_page if total_positive > 0 else 1
                    page_positive = st.number_input("Pozitif örnekler - Sayfa", min_value=1, max_value=num_pages_positive, value=1, key="page_positive")
                    
                    start_idx_positive = (page_positive - 1) * num_per_page
                    end_idx_positive = start_idx_positive + num_per_page
                    positive_page = positive_responses.iloc[start_idx_positive:end_idx_positive]
                    
                    st.caption(f"Toplam {total_positive} pozitif örnek bulundu. Sayfa {page_positive}/{num_pages_positive}")
                    
                    for idx, row in positive_page.iterrows():
                        st.markdown(f"**{row['text']}**")
                        topics_label = row['topics_raw'] if pd.notna(row['topics_raw']) else "Yok"
                        
                        # Get free topics for this response
                        free_long = df.attrs.get("free_long", pd.DataFrame())
                        free_topics_label = "Yok"
                        if not free_long.empty and "response_index" in row and "question_num" in row:
                            response_idx = row["response_index"]
                            question_num = row["question_num"]
                            matching_free = free_long[
                                (free_long["response_index"] == response_idx) & 
                                (free_long["question_num"] == question_num)
                            ]
                            if not matching_free.empty:
                                free_topics_list = matching_free["free_topic"].dropna().unique().tolist()
                                if free_topics_list:
                                    free_topics_label = ", ".join(free_topics_list)
                        
                        base_caption = f"Sentiment: {row['sentiment_raw']} | Topics: {topics_label} | Free Topics: {free_topics_label}"
                        
                        # Add ID columns to caption
                        id_parts = []
                        # Add InviteeFullName first if available
                        if "InviteeFullName" in row and pd.notna(row["InviteeFullName"]) and str(row["InviteeFullName"]).strip() != "":
                            id_parts.append(f"İsim: {row['InviteeFullName']}")
                        elif "AgentName" in row and pd.notna(row["AgentName"]) and str(row["AgentName"]).strip() != "":
                            id_parts.append(f"İsim: {row['AgentName']}")
                        
                        mapping = [
                            ("Acente Bölge", "Bölge"),
                            ("Acente İli", "İl"),
                            ("enSegmenti", "Segment"),
                            ("Sınıf", "Sınıf"),
                            ("Grup", "Grup"),
                            ("Harf Skoru", "Harf"),
                        ]
                        for col, label in mapping:
                            if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
                                id_parts.append(f"{label}: {row[col]}")
                        
                        if id_parts:
                            st.caption(base_caption + " | " + " | ".join(id_parts))
                        else:
                            st.caption(base_caption)
                        st.divider()
                else:
                    st.info("Pozitif örnek bulunamadı.")
    
                # Negatif örnekler
                st.subheader("⚠️ Negatif Örnekler")
                negative_responses = unique_responses[unique_responses["sentiment"] == "negative"].copy()
                
                if not negative_responses.empty:
                    total_negative = len(negative_responses)
                    num_pages_negative = (total_negative + num_per_page - 1) // num_per_page if total_negative > 0 else 1
                    page_negative = st.number_input("Negatif örnekler - Sayfa", min_value=1, max_value=num_pages_negative, value=1, key="page_negative")
                    
                    start_idx_negative = (page_negative - 1) * num_per_page
                    end_idx_negative = start_idx_negative + num_per_page
                    negative_page = negative_responses.iloc[start_idx_negative:end_idx_negative]
                    
                    st.caption(f"Toplam {total_negative} negatif örnek bulundu. Sayfa {page_negative}/{num_pages_negative}")
                    
                    for idx, row in negative_page.iterrows():
                        st.markdown(f"**{row['text']}**")
                        topics_label = row['topics_raw'] if pd.notna(row['topics_raw']) else "Yok"
                        
                        # Get free topics for this response
                        free_long = df.attrs.get("free_long", pd.DataFrame())
                        free_topics_label = "Yok"
                        if not free_long.empty and "response_index" in row and "question_num" in row:
                            response_idx = row["response_index"]
                            question_num = row["question_num"]
                            matching_free = free_long[
                                (free_long["response_index"] == response_idx) & 
                                (free_long["question_num"] == question_num)
                            ]
                            if not matching_free.empty:
                                free_topics_list = matching_free["free_topic"].dropna().unique().tolist()
                                if free_topics_list:
                                    free_topics_label = ", ".join(free_topics_list)
                        
                        base_caption = f"Sentiment: {row['sentiment_raw']} | Topics: {topics_label} | Free Topics: {free_topics_label}"
                        
                        # Add ID columns to caption
                        id_parts = []
                        # Add InviteeFullName first if available
                        if "InviteeFullName" in row and pd.notna(row["InviteeFullName"]) and str(row["InviteeFullName"]).strip() != "":
                            id_parts.append(f"İsim: {row['InviteeFullName']}")
                        elif "AgentName" in row and pd.notna(row["AgentName"]) and str(row["AgentName"]).strip() != "":
                            id_parts.append(f"İsim: {row['AgentName']}")
                        
                        mapping = [
                            ("Acente Bölge", "Bölge"),
                            ("Acente İli", "İl"),
                            ("enSegmenti", "Segment"),
                            ("Sınıf", "Sınıf"),
                            ("Grup", "Grup"),
                            ("Harf Skoru", "Harf"),
                        ]
                        for col, label in mapping:
                            if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
                                id_parts.append(f"{label}: {row[col]}")
                        
                        if id_parts:
                            st.caption(base_caption + " | " + " | ".join(id_parts))
                        else:
                            st.caption(base_caption)
                        st.divider()
                else:
                    st.info("Negatif örnek bulunamadı.")
                    
    # TAB 6: Likert Analizleri
    with tab6:
        st.header("📊 Likert Analizleri")
        
        # Check for likert error
        likert_error = df.attrs.get("likert_error")
        if likert_error:
            st.error(likert_error)
            st.stop()
        
        # Terim Açıklamaları
        with st.expander("📖 Terim Açıklamaları", expanded=False):
            st.markdown("""
            **Likert Analizi Terimleri:**
            - **Likert Ölçeği**: Adını bu ölçeğin oluşturucusu olan Amerikalı sosyal bilimci Rensis Likert'ten alan Likert ölçekleri ve analizi psikometrik terimlerdir; fikirleri, algıları ve davranışları ölçmenin en güvenilir yollarından biri olarak oldukça sık tercih edilir.
            - **Skor Dağılımı**: Her skor değerinin (1-5) kaç kez seçildiğinin gösterimi
            - **Ortalama Skor**: Tüm yanıtların sayısal ortalaması
            - **Kullanmıyorum Oranı**: "Kullanmıyorum (6)" seçeneğini işaretleyenlerin oranı
            - **Grup Ortalaması**: Belirli bir soru grubunun ortalama skoru
            """)
        
        likert_groups_ui = df.attrs.get("likert_groups_ui", {})
        likert_question_texts = df.attrs.get("likert_question_texts", {})
        likert_numeric_cols = df.attrs.get("likert_numeric_cols", {})
        LIKERT_GROUPS_ATTR = df.attrs.get("likert_groups", LIKERT_GROUPS)
        
        if not likert_numeric_cols:
            st.warning("Likert soru verisi bulunamadı.")
            return
        
        # Use likert_groups_ui if available, otherwise fall back to LIKERT_GROUPS_ATTR
        if likert_groups_ui:
            group_options = ["(Tümü)"] + list(likert_groups_ui.keys())
        else:
            group_options = ["(Tümü)"] + list(LIKERT_GROUPS_ATTR.keys())
        
        selected_group = st.selectbox("Grup Seçin", options=group_options)
        
        # Build available_questions based on selected group
        available_questions = []
        
        if selected_group == "(Tümü)":
            # Iterate over all entries in likert_numeric_cols
            for num, col_name in likert_numeric_cols.items():
                question_text = likert_question_texts.get(num, f"Soru {num}")
                available_questions.append((num, question_text, col_name))
        else:
            # Get question numbers from selected group
            if likert_groups_ui and selected_group in likert_groups_ui:
                question_nums = likert_groups_ui[selected_group]
            elif selected_group in LIKERT_GROUPS_ATTR:
                question_nums = LIKERT_GROUPS_ATTR[selected_group]
            else:
                question_nums = []
            
            for num in question_nums:
                if num in likert_numeric_cols:
                    question_text = likert_question_texts.get(num, f"Soru {num}")
                    col_name = likert_numeric_cols[num]
                    available_questions.append((num, question_text, col_name))
        
        if not available_questions:
            st.info("Seçilebilir soru bulunamadı.")
        else:
            # Build question dropdown labels
            question_options_likert = {}
            for num, question_text, col_name in available_questions:
                # Truncate question text to 80 characters
                truncated_text = question_text[:80] + "..." if len(question_text) > 80 else question_text
                label = f"{num} - {truncated_text}"
                question_options_likert[label] = (num, col_name)
            
            selected_question_label = st.selectbox(
                "Soru Seçin",
                options=list(question_options_likert.keys())
            )
            
            selected_num, selected_col = question_options_likert[selected_question_label]
            
            # Soru dağılımı
            st.subheader(f"Soru {selected_num} - Dağılım")
            distribution = df[selected_col].value_counts().sort_index()
            distribution_pct = df[selected_col].value_counts(normalize=True).sort_index() * 100
            
            dist_df = pd.DataFrame({
                "Skor": distribution.index,
                "Frekans": distribution.values,
                "Yüzde (%)": distribution_pct.values.round(2)
            })
            st.dataframe(dist_df, use_container_width=True)
            
            # Özet istatistikler
            st.subheader(f"Soru {selected_num} - Özet İstatistikler")
            valid_data = df[selected_col].dropna()
            stats = {
                "Ortalama": valid_data.mean(),
                "Medyan": valid_data.median(),
                "Standart Sapma": valid_data.std(),
                "Min": valid_data.min(),
                "Max": valid_data.max(),
                "Geçerli Yanıt": len(valid_data),
                "Kullanmıyorum Oranı": ((len(df) - len(valid_data)) / len(df) * 100) if len(df) > 0 else 0
            }
            stats_df = pd.DataFrame([stats])
            st.dataframe(stats_df, use_container_width=True)
            
            # Dağılım grafiği
            st.subheader(f"Soru {selected_num} - Dağılım Grafiği")
            fig_likert = px.bar(
                dist_df,
                x='Skor',
                y='Frekans',
                title=f'Soru {selected_num} - Dağılım',
                labels={'Skor': 'Skor', 'Frekans': 'Frekans'}
            )
            fig_likert.update_layout(height=300)
            st.plotly_chart(fig_likert, use_container_width=True)
            
            # Grup ortalamaları
            if selected_group == "(Tümü)":
                st.subheader("Tüm Gruplar - Ortalama Skorlar")
                group_avgs = []
                
                # Use likert_groups_ui if available to get group names from konular sheet
                if likert_groups_ui:
                    # Calculate average for each group from konular sheet
                    for group_name_ui in likert_groups_ui.keys():
                        question_nums = likert_groups_ui[group_name_ui]
                        # Get numeric columns for questions in this group
                        numeric_cols = [likert_numeric_cols.get(num) for num in question_nums if num in likert_numeric_cols]
                        numeric_cols = [col for col in numeric_cols if col and col in df.columns]
                        
                        if numeric_cols:
                            # Calculate row-wise mean for this group
                            avg_score = df[numeric_cols].mean(axis=1, skipna=True).mean()
                            group_avgs.append({
                                "Grup": group_name_ui,
                                "Ortalama Skor": avg_score
                            })
                else:
                    # Fallback: try to find Likert average columns dynamically
                    likert_avg_cols = [col for col in df.columns if col.endswith("_Likert_Avg")]
                    for avg_col in likert_avg_cols:
                        # Skip NPS and Öneri columns
                        if "NPS" in avg_col or "Öneri" in avg_col:
                            continue
                        group_name = avg_col.replace("_Likert_Avg", "")
                        avg_score = df[avg_col].mean()
                        group_avgs.append({
                            "Grup": group_name,
                            "Ortalama Skor": avg_score
                        })
                
                if group_avgs:
                    group_avg_df = pd.DataFrame(group_avgs)
                    group_avg_df = group_avg_df.sort_values("Ortalama Skor", ascending=True)
                    
                    fig_all_groups = px.bar(
                        group_avg_df,
                        x='Ortalama Skor',
                        y='Grup',
                        orientation='h',
                        title='Tüm Gruplar - Ortalama Skorlar',
                        labels={'Ortalama Skor': 'Ortalama Skor', 'Grup': 'Grup'}
                    )
                    fig_all_groups.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_all_groups, use_container_width=True)
            else:
                st.subheader(f"{selected_group} Grubu - Soru Ortalamaları")
                group_questions = []
                
                # Get question numbers from likert_groups_ui if available, otherwise from LIKERT_GROUPS_ATTR
                if likert_groups_ui and selected_group in likert_groups_ui:
                    question_nums = likert_groups_ui[selected_group]
                elif selected_group in LIKERT_GROUPS_ATTR:
                    question_nums = LIKERT_GROUPS_ATTR[selected_group]
                else:
                    question_nums = []
                
                for num in question_nums:
                    if num in likert_numeric_cols:
                        avg_score = df[likert_numeric_cols[num]].mean()
                        question_text = likert_question_texts.get(num, f"Soru {num}")
                        # Truncate for display
                        display_text = question_text[:60] + "..." if len(question_text) > 60 else question_text
                        group_questions.append({
                            "Soru": f"{num} - {display_text}",
                            "Ortalama Skor": avg_score
                        })
                
                if group_questions:
                    group_df = pd.DataFrame(group_questions)
                    group_df = group_df.sort_values("Ortalama Skor", ascending=True)
                    
                    fig_group = px.bar(
                        group_df,
                        x='Ortalama Skor',
                        y='Soru',
                        orientation='h',
                        title=f'{selected_group} Grubu - Soru Ortalamaları',
                        labels={'Ortalama Skor': 'Ortalama Skor', 'Soru': 'Soru'}
                    )
                    fig_group.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_group, use_container_width=True)
    
    # TAB 7: Likert vs Açık Uç
    with tab7:
        st.header("🔗 Likert vs Açık Uç Karşılaştırması")
        
        # Check for likert error
        likert_error = df.attrs.get("likert_error")
        if likert_error:
            st.error(likert_error)
            st.info("Likert vs Açık Uç karşılaştırmaları için önce 'likert+NPS' sheet'ini oluşturmanız gerekiyor.")
            st.stop()
        
        # Terim Açıklamaları
        with st.expander("📖 Terim Açıklamaları", expanded=False):
            st.markdown("""
            **Likert vs Açık Uç Terimleri:**
            - **Spearman Korelasyon**: Sıralı veriler için kullanılan korelasyon yöntemi
            - **Segmentasyon**: Likert skorları ve sentiment skorlarına göre katılımcıların gruplandırılması
            - **Uyumlu Pozitif**: Hem Likert hem sentiment skorları yüksek olanlar
            - **Kibar ama Mutsuz**: Likert skoru yüksek ama sentiment skoru düşük olanlar
            - **Açık Risk**: Hem Likert hem sentiment skorları düşük olanlar
            - **Potansiyel**: Likert skoru düşük ama sentiment skoru yüksek olanlar
            """)
        
        # Global karşılaştırma
        st.subheader("Global Karşılaştırma")
        
        # Find the correct "Genel" Likert average column
        genel_likert_col = None
        for col in df.columns:
            if col.endswith("_Likert_Avg") and ("Genel" in col or "genel" in col.lower()):
                # Prefer "Genel Memnuniyet" if available
                if "Genel Memnuniyet" in col:
                    genel_likert_col = col
                    break
                elif genel_likert_col is None:
                    genel_likert_col = col
        
        if genel_likert_col is None or "OpenSentiment_Avg" not in df.columns:
            st.warning(f"Gerekli kolonlar bulunamadı (Genel Likert: {genel_likert_col}, OpenSentiment_Avg).")
        else:
            comparison_df = df[[genel_likert_col, "OpenSentiment_Avg"]].copy()
            if "AgentName" in df.columns:
                comparison_df["AgentName"] = df["AgentName"]
            elif "InviteeFullName" in df.columns:
                comparison_df["AgentName"] = df["InviteeFullName"]
            else:
                comparison_df["AgentName"] = "Bilinmiyor"
            
            # Add ID columns for hover
            id_cols_for_hover = [
                "AgentName", "InviteeFullName",
                "Acente Bölge", "Acente İli",
                "Acente Açılış Tarihi",
                "enSegmenti", "Sınıf", "Grup", "Harf Skoru",
            ]
            for col in id_cols_for_hover:
                if col in df.columns and col not in comparison_df.columns:
                    comparison_df[col] = df[col]
            
            hover_cols = [c for c in id_cols_for_hover if c in comparison_df.columns]
            
            comparison_df = comparison_df.dropna(subset=[genel_likert_col, "OpenSentiment_Avg"])
            
            if not comparison_df.empty:
                fig_scatter = px.scatter(
                    comparison_df,
                    x=genel_likert_col,
                    y='OpenSentiment_Avg',
                    hover_data=hover_cols,
                    title='Genel Likert vs Açık Uç Sentiment',
                    labels={genel_likert_col: 'Genel Likert Ortalaması (1-5)', 'OpenSentiment_Avg': 'Açık Uç Sentiment Ortalaması (-1 to +1)'}
                )
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                correlation = comparison_df[genel_likert_col].corr(comparison_df["OpenSentiment_Avg"], method="spearman")
                st.write(f"**Spearman Korelasyon Katsayısı:** {correlation:.3f}")
            else:
                st.info("Karşılaştırma için yeterli veri bulunamadı.")
        
        # Tematik karşılaştırma
        st.subheader("Tematik Karşılaştırma")
        
        # Find all Likert average columns and matching OpenSentiment columns
        likert_avg_cols = [col for col in df.columns if col.endswith("_Likert_Avg")]
        open_sentiment_cols = [col for col in df.columns if col.endswith("_OpenSentiment")]
        
        # Build thematic options by matching Likert and OpenSentiment columns
        thematic_options = []
        thematic_mapping = {}  # Maps theme name to (likert_col, sentiment_col)
        
        for likert_col in likert_avg_cols:
            # Extract theme name from Likert column (remove "_Likert_Avg")
            theme_base = likert_col.replace("_Likert_Avg", "")
            
            # Skip "Genel" and "NPS" related columns
            if "Genel" in theme_base or "NPS" in theme_base or "Öneri" in theme_base:
                continue
            
            # Try to find matching OpenSentiment column
            for sentiment_col in open_sentiment_cols:
                sentiment_base = sentiment_col.replace("_OpenSentiment", "")
                
                # Match if the base names are similar (exact match or one contains the other)
                if theme_base == sentiment_base or theme_base in sentiment_base or sentiment_base in theme_base:
                    thematic_options.append(theme_base)
                    thematic_mapping[theme_base] = (likert_col, sentiment_col)
                    break
        
        # Remove duplicates while preserving order
        thematic_options = list(dict.fromkeys(thematic_options))
        
        if not thematic_options:
            st.info("Tematik karşılaştırma için uygun Likert ve açık uç sentiment kolonları bulunamadı.")
        else:
            selected_theme = st.selectbox("Tema Seçin", options=thematic_options)
            
            if selected_theme not in thematic_mapping:
                st.warning(f"Seçilen tema için kolonlar bulunamadı: {selected_theme}")
            else:
                likert_col, sentiment_col = thematic_mapping[selected_theme]
                
                if likert_col not in df.columns or sentiment_col not in df.columns:
                    st.warning(f"Gerekli kolonlar bulunamadı ({likert_col}, {sentiment_col}).")
                else:
                    theme_df = df[[likert_col, sentiment_col]].copy()
                    if "AgentName" in df.columns:
                        theme_df["AgentName"] = df["AgentName"]
                    elif "InviteeFullName" in df.columns:
                        theme_df["AgentName"] = df["InviteeFullName"]
                    else:
                        theme_df["AgentName"] = "Bilinmiyor"
                    
                    # Add ID columns for hover
                    id_cols_for_hover = [
                        "AgentName", "InviteeFullName",
                        "Acente Bölge", "Acente İli",
                        "Acente Açılış Tarihi",
                        "enSegmenti", "Sınıf", "Grup", "Harf Skoru",
                    ]
                    for col in id_cols_for_hover:
                        if col in df.columns and col not in theme_df.columns:
                            theme_df[col] = df[col]
                    
                    hover_cols = [c for c in id_cols_for_hover if c in theme_df.columns]
                    
                    theme_df = theme_df.dropna(subset=[likert_col, sentiment_col])
                    
                    if not theme_df.empty:
                        fig_theme = px.scatter(
                            theme_df,
                            x=likert_col,
                            y=sentiment_col,
                            hover_data=hover_cols,
                            title=f'{selected_theme} Likert vs Açık Uç Sentiment',
                            labels={likert_col: f'{selected_theme} Likert Ortalaması (1-5)', sentiment_col: f'{selected_theme} Açık Uç Sentiment (-1 to +1)'}
                        )
                        fig_theme.update_layout(height=400)
                        st.plotly_chart(fig_theme, use_container_width=True)
                        
                        correlation = theme_df[likert_col].corr(theme_df[sentiment_col], method="spearman")
                        st.write(f"**Spearman Korelasyon Katsayısı:** {correlation:.3f}")
                        
                        # Segmentasyon
                        st.subheader("Segmentasyon Analizi")
                        likert_threshold = st.slider("Likert Eşik Değeri", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
                        sentiment_threshold = st.slider("Sentiment Eşik Değeri", min_value=-1.0, max_value=1.0, value=0.0, step=0.1)
                        
                        theme_df["Segment"] = "Diğer"
                        theme_df.loc[
                            (theme_df[likert_col] >= likert_threshold) & (theme_df[sentiment_col] >= sentiment_threshold),
                            "Segment"
                        ] = "Uyumlu pozitif"
                        theme_df.loc[
                            (theme_df[likert_col] >= likert_threshold) & (theme_df[sentiment_col] < sentiment_threshold),
                            "Segment"
                        ] = "Kibar ama mutsuz"
                        theme_df.loc[
                            (theme_df[likert_col] < likert_threshold) & (theme_df[sentiment_col] < sentiment_threshold),
                            "Segment"
                        ] = "Açık risk"
                        theme_df.loc[
                            (theme_df[likert_col] < likert_threshold) & (theme_df[sentiment_col] >= sentiment_threshold),
                            "Segment"
                        ] = "Potansiyel"
                        
                        segment_counts = theme_df["Segment"].value_counts()
                        st.dataframe(segment_counts.reset_index().rename(columns={"index": "Segment", "Segment": "Sayı"}), use_container_width=True)
                    else:
                        st.info("Tematik karşılaştırma için yeterli veri bulunamadı.")
    
    # TAB 8: Wordcloud
    with tab8:
        st.header("☁️ Free Topics Wordcloud")
        
        with st.expander("📖 Terim Açıklamaları", expanded=False):
            st.markdown(
                "- **Free topics**: OpenAI API'den serbest metin olarak gelen konu etiketleri.\n"
                "- **Açık uçlu soru filtresi**: Belirli bir açık uçlu sorunun free_topics dağılımını görebilirsiniz."
            )
        
        free_long = df.attrs.get("free_long")
        open_question_meta = df.attrs.get("open_question_meta", [])
        
        if free_long is None or free_long.empty:
            st.info("Free topics verisi bulunamadı.")
        else:
            # Build list of question labels from free_long
            if not free_long.empty and "question_short" in free_long.columns:
                question_labels = sorted(free_long["question_short"].unique())
            else:
                question_labels = sorted({m.get("question_short", f"Soru {m['num']}") for m in open_question_meta if m.get("question_short")})
            question_options = ["(Tümü)"] + question_labels
            
            # Multiselect for questions
            selected_questions = st.multiselect(
                "Açık uçlu soru filtresi",
                options=question_options,
                default=["(Tümü)"]
            )
            
            # Filter free_long
            if "(Tümü)" in selected_questions or not selected_questions:
                filtered = free_long.copy()
            else:
                filtered = free_long[free_long["question_short"].isin(selected_questions)].copy()
            
            if filtered.empty:
                st.info("Seçilen filtreler için free topics verisi bulunamadı.")
            else:
                # Build pivot dataframe (question_col x free_topic counts)
                # Use question_col instead of question_short for full question names
                if "question_col" in filtered.columns:
                    pivot_df = (
                        filtered.groupby(["question_col", "free_topic"])
                        .size()
                        .reset_index(name="count")
                        .sort_values("count", ascending=False)
                    )
                    # Rename question_col to "Soru" for display
                    pivot_df = pivot_df.rename(columns={"question_col": "Soru", "free_topic": "Free Topic", "count": "Frekans"})
                else:
                    # Fallback to question_short if question_col not available
                    pivot_df = (
                        filtered.groupby(["question_short", "free_topic"])
                        .size()
                        .reset_index(name="count")
                        .sort_values("count", ascending=False)
                    )
                    pivot_df = pivot_df.rename(columns={"question_short": "Soru", "free_topic": "Free Topic", "count": "Frekans"})
                
                st.subheader("Açık Uçlu Soru x Free Topics Pivotu")
                st.dataframe(pivot_df, use_container_width=True, height=350)
                
                # Generate WordCloud from filtered dataframe
                freq = filtered["free_topic"].dropna()
                freq = freq[freq != ""]
                word_counts = freq.value_counts()
                
                if word_counts.empty:
                    st.info("Free topics verisi bulunamadı.")
                else:
                    # Slider for minimum frequency
                    min_freq = st.slider("Minimum Frekans", min_value=1, max_value=int(word_counts.max()), value=1, key="wordcloud_min_freq")
                    wc_freq = word_counts[word_counts >= min_freq]
                    
                    if not wc_freq.empty:
                        try:
                            wc = WordCloud(width=1600, height=800, background_color="white")
                            wc = wc.generate_from_frequencies(wc_freq.to_dict())
                            # Convert to image for Streamlit
                            fig, ax = plt.subplots(figsize=(16, 8))
                            ax.imshow(wc, interpolation="bilinear")
                            ax.axis("off")
                            ax.set_title("Free Topics Wordcloud", fontsize=16, pad=20)
                            # Save to buffer
                            buf = io.BytesIO()
                            plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                            buf.seek(0)
                            st.image(buf, use_container_width=True)
                            plt.close(fig)
                        except Exception as e:
                            st.error(f"Wordcloud oluşturulurken hata oluştu: {str(e)}")
                    else:
                        st.info(f"Seçilen minimum frekans ({min_freq}) için free topics bulunamadı.")
    
    # TAB 9: Ham Veri
    with tab9:
        st.header("📋 Ham Veri Görüntüleme")
        
        # Terim Açıklamaları
        with st.expander("📖 Terim Açıklamaları", expanded=False):
            st.markdown("""
            **Ham Veri Terimleri:**
            - **Ana DataFrame**: Tüm anket verilerini içeren ana veri tablosu
            - **open_long**: Açık uçlu yanıtların uzun format (long format) halinde düzenlenmiş hali
            - **free_long**: Free topics ve Topics verilerinin uzun format halinde düzenlenmiş hali
            - **response_index**: Her yanıtın ana DataFrame'deki orijinal satır indeksi
            """)
        
        data_option = st.selectbox(
            "Gösterilecek Veri",
            options=["Ana DataFrame", "open_long", "free_long"]
        )
        
        if data_option == "Ana DataFrame":
            st.dataframe(df, use_container_width=True, height=400)
        elif data_option == "open_long":
            if not open_long.empty:
                # Remove question_short, sentiment_raw, topics_raw columns
                display_cols = [col for col in open_long.columns if col not in ["question_short", "sentiment_raw", "topics_raw"]]
                st.dataframe(open_long[display_cols], use_container_width=True, height=400)
            else:
                st.info("open_long verisi bulunamadı.")
        elif data_option == "free_long":
            if not free_long.empty:
                # Remove question_short, show question_col instead
                display_cols_free = []
                for col in free_long.columns:
                    if col == "question_short":
                        continue  # Skip question_short
                    display_cols_free.append(col)
                # Ensure question_col is included if it exists
                if "question_col" in free_long.columns and "question_col" not in display_cols_free:
                    display_cols_free.insert(display_cols_free.index("question_num") + 1 if "question_num" in display_cols_free else 0, "question_col")
                # Rename 'topic' column to 'topics' for display
                display_df = free_long[display_cols_free].copy()
                if 'topic' in display_df.columns:
                    display_df = display_df.rename(columns={'topic': 'topics'})
                st.dataframe(display_df, use_container_width=True, height=400)
            else:
                st.info("free_long verisi bulunamadı.")


if __name__ == "__main__":
    main()
