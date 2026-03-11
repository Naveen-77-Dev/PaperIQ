import streamlit as st
import fitz  # PyMuPDF
import re
from collections import Counter
import math
import sqlite3
import hashlib
import time
import pandas as pd
import heapq
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
from spacy.lang.en import English

# --- SPACY MODEL LOADING ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PaperIQ",
    page_icon="■",
    layout="wide",
)

# --- AI MODEL LOADING (Transformers) ---
try:
    from transformers import pipeline
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    pipeline = None

@st.cache_resource
def load_summarizer():
    if AI_AVAILABLE:
        try:
            return pipeline("summarization", model="t5-small")
        except Exception:
            return None
    return None

if AI_AVAILABLE:
    summarizer = load_summarizer()
else:
    summarizer = None

# --- DATABASE ENGINE ---
DB_NAME = "paperiq.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    email TEXT PRIMARY KEY,
                    fullname TEXT,
                    password TEXT,
                    role TEXT,
                    security_question TEXT,
                    security_answer TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS upload_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_email TEXT,
                    file_name TEXT,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    page_count INTEGER,
                    word_count INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS saved_papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_email TEXT,
                    file_name TEXT,
                    summary_abstract TEXT,
                    saved_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

def run_query(query, params=(), fetch_one=False, fetch_all=False):
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, params)
        if fetch_one:
            result = cursor.fetchone()
        elif fetch_all:
            result = cursor.fetchall()
        else:
            conn.commit()
            result = True
        conn.close()
        return result
    except Exception:
        return None

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# --- SESSION STATE INITIALIZATION ---
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = 'Student'
if 'user_name' not in st.session_state:
    st.session_state.user_name = "User"
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""
if 'menu_selection' not in st.session_state:
    st.session_state.menu_selection = 'Dashboard'
if 'analyses' not in st.session_state:
    st.session_state.analyses = {}
if 'current_filename' not in st.session_state:
    st.session_state.current_filename = ""
if 'user_initial' not in st.session_state:
    st.session_state.user_initial = "U"
if 'summary_length' not in st.session_state:
    st.session_state.summary_length = "Medium"

# Forgot password states
if 'fp_step' not in st.session_state:
    st.session_state.fp_step = 1
if 'fp_email' not in st.session_state:
    st.session_state.fp_email = ""
if 'fp_sq' not in st.session_state:
    st.session_state.fp_sq = ""

# --- DARK THEME CSS (permanent) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    :root {
        --bg-color: #050505;
        --card-bg: #111111;
        --text-color: #FFFFFF;
        --text-dim: #A0A0A0;
        --border: #333333;
        --button-bg: #1E1E1E;
        --primary-blue: #0056b3;
        --input-bg: #0F0F0F;
        --expander-bg: #1A1A1A;
        --expander-content-bg: #0A0A0A;
        --stat-box-bg: #161616;
        --logout-bg: #330000;
        --logout-hover: #FF0000;
        --logout-text: #FF4444;
        --file-uploader-bg: #0F0F0F;
        --file-uploader-text: #FFFFFF;
        --file-uploader-border: #333333;
    }

    .stApp {
        background-color: var(--bg-color);
        font-family: 'Inter', sans-serif;
        color: var(--text-color);
    }

    h1, h2, h3, h4 { color: var(--text-color) !important; font-weight: 800; }
    p, label, span, div { color: var(--text-dim); }

    /* --- Navigation buttons styled by key --- */
    div.stButton > button[key="nav_dash"],
    div.stButton > button[key="nav_saved"],
    div.stButton > button[key="nav_hist"] {
        background-color: var(--button-bg);
        color: var(--text-color);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        width: 100%;
    }
    div.stButton > button[key="nav_dash"]:hover,
    div.stButton > button[key="nav_saved"]:hover,
    div.stButton > button[key="nav_hist"]:hover {
        background-color: var(--border);
        border-color: var(--text-color);
    }

    /* Logout button */
    div.stButton > button[key="logout_btn"] {
        background-color: var(--logout-bg) !important;
        color: var(--logout-text) !important;
        border: 1px solid var(--logout-text) !important;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
    div.stButton > button[key="logout_btn"]:hover {
        background-color: var(--logout-hover) !important;
    }

    /* Profile circle button */
    div.stButton > button[key="profile_circle"] {
        background-color: var(--primary-blue);
        color: white;
        border-radius: 50%;
        width: 45px;
        height: 45px;
        padding: 0;
        font-size: 18px;
        font-weight: 800;
        border: 2px solid var(--text-color);
        display: flex;
        align-items: center;
        justify-content: center;
        margin-left: auto;
        margin-right: auto;
    }
    div.stButton > button[key="profile_circle"]:hover {
        background-color: #004494;
        transform: scale(1.1);
    }

    /* Analyze button */
    div.stButton > button[key="analyze_btn"] {
        background: linear-gradient(90deg, var(--primary-blue) 0%, #004494 100%);
        color: white;
        border: none;
        padding: 15px;
        font-size: 1.1rem;
        border-radius: 8px;
        margin-top: 20px;
        width: 100%;
    }
    div.stButton > button[key="analyze_btn"]:hover {
        opacity: 0.9;
        box-shadow: 0 0 15px rgba(0,86,179,0.5);
    }

    /* File uploader */
    .stFileUploader {
        background-color: var(--file-uploader-bg);
        color: var(--file-uploader-text);
        border: 1px dashed var(--file-uploader-border);
        border-radius: 8px;
        padding: 20px;
    }
    .stFileUploader label {
        color: var(--file-uploader-text) !important;
    }
    .stFileUploader small {
        color: var(--text-dim) !important;
    }
    .stFileUploader button {
        background-color: var(--primary-blue);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 5px 10px;
    }

    /* Info cards */
    .info-card {
        background-color: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 20px;
    }
    .info-title { color: var(--text-color); font-size: 1.2rem; font-weight: 700; margin-bottom: 10px; }
    .info-text { color: var(--text-dim); font-size: 0.95rem; line-height: 1.6; }

    /* Input fields */
    input[type="text"], input[type="password"], textarea {
        background-color: var(--input-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: var(--expander-bg) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-color) !important;
        border-radius: 8px !important;
    }
    div[data-testid="stExpander"] details > div {
        background-color: var(--expander-content-bg);
        border: 1px solid var(--border);
        border-top: none;
        padding: 20px;
    }

    /* Stat boxes */
    .stat-box { background: var(--stat-box-bg); padding: 15px; border-radius: 8px; text-align: center; border: 1px solid var(--border); }
    .stat-val { font-size: 1.5rem; font-weight: 800; color: var(--text-color); }
    .stat-lbl { font-size: 0.8rem; text-transform: uppercase; color: var(--text-dim); }

    /* Metrics (st.metric) */
    div[data-testid="stMetricValue"] {
        color: var(--text-color) !important;
    }
    div[data-testid="stMetricLabel"] {
        color: var(--text-dim) !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: var(--text-color) !important;
    }
    div[data-testid="stTabs"] {
        background-color: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px;
    }

    /* Select slider */
    div[data-testid="stSlider"] {
        color: var(--text-color);
    }

    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- AUTH LOGIC ---
def register_user(email, fullname, password, role, sq, sa):
    exists = run_query("SELECT email FROM users WHERE email = ?", (email,), fetch_one=True)
    if exists:
        return False, "Email already registered."
    result = run_query("INSERT INTO users (email, fullname, password, role, security_question, security_answer) VALUES (?, ?, ?, ?, ?, ?)",
                       (email, fullname, hash_password(password), role, sq, sa.lower().strip()))
    return (True, "Account created!") if result else (False, "Database Error")

def login_user(email, password):
    return run_query("SELECT * FROM users WHERE email = ? AND password = ?", (email, hash_password(password)), fetch_one=True)

def get_security_question(email):
    res = run_query("SELECT security_question FROM users WHERE email = ?", (email,), fetch_one=True)
    return res['security_question'] if res else None

def verify_security_answer(email, answer):
    res = run_query("SELECT email FROM users WHERE email = ? AND security_answer = ?", (email, answer.lower().strip()), fetch_one=True)
    return True if res else False

def update_password(email, new_password):
    return run_query("UPDATE users SET password = ? WHERE email = ?", (hash_password(new_password), email))

# --- SAFE TEXT ENCODING FOR PDF EXPORT (prevents Unicode crashes) ---
def safe_text(text):
    """Convert any text to latin-1 safely, replacing unsupported characters."""
    if not isinstance(text, str):
        text = str(text)
    return text.encode('latin-1', errors='replace').decode('latin-1')

# --- EXPORT FUNCTIONS ---
def generate_markdown(engine, filename):
    """Generate a Markdown string with all analysis details for a paper."""
    md = f"# PaperIQ Analysis Report: {filename}\n\n"
    md += f"## Final Composite Score: {engine.scores['Composite']:.2f}/100\n\n"
    md += "### Scores\n"
    for key, val in engine.scores.items():
        if key != "Composite":
            md += f"- **{key}**: {val:.2f}/100\n"
    md += "\n### Text Statistics\n"
    md += f"- Words: {engine.stats['word_count']}\n"
    md += f"- Sentences: {engine.stats['sentence_count']}\n"
    md += f"- Avg Sentence Length: {engine.stats['avg_sentence_len']:.2f}\n"
    md += f"- Avg Word Length: {engine.stats['avg_word_len']:.2f}\n"
    md += f"- Vocabulary Diversity: {engine.stats.get('vocab_diversity', 0):.2f}\n"
    md += f"- Complex Word Ratio: {engine.stats.get('complex_word_ratio', 0):.2f}\n\n"
    md += f"### Domain\n{engine.domain}\n\n"
    md += "### Keywords\n"
    if engine.present_keywords:
        md += f"- Present: {', '.join(engine.present_keywords)}\n"
    if engine.missing_keywords:
        md += f"- Missing: {', '.join(engine.missing_keywords)}\n"
    if not engine.present_keywords and not engine.missing_keywords:
        md += "- No keywords provided.\n"
    md += "\n### Top 10 Words (excluding stopwords)\n"
    for word, count in engine.word_freq[:10]:
        md += f"- {word}: {count}\n"
    md += "\n### Sentiment\n"
    sentiment = engine.sentiment
    if sentiment > 0.05:
        label = "Positive"
    elif sentiment < -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    md += f"- Polarity: {sentiment:.2f} ({label})\n\n"
    md += "### Section Summaries (Medium)\n"
    for title in ['Abstract', 'Introduction', 'Methodology', 'Results', 'Conclusion']:
        summary = engine.ai_summaries.get(title, {}).get('Medium', 'N/A')
        md += f"**{title}**\n\n{summary}\n\n"
    if engine.issues:
        md += "### Long Sentences (>30 words)\n"
        for i, s in enumerate(engine.issues[:5]):
            md += f"{i+1}. {s}\n"
        if len(engine.issues) > 5:
            md += f"\n... and {len(engine.issues)-5} more.\n"
    suggestions = engine._generate_suggestions()
    if suggestions:
        md += "\n### Vocabulary Suggestions\n"
        for s in suggestions:
            md += f"- {s}\n"
    return md

def generate_combined_pdf(analyses):
    """Generate a PDF report containing all papers."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for filename, engine in analyses.items():
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=safe_text(f"PaperIQ Analysis Report: {filename}"), ln=1, align='C')
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt=safe_text(f"Final Composite Score: {engine.scores['Composite']:.2f}/100"), ln=1)
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=safe_text("Scores:"), ln=1)
        pdf.set_font("Arial", size=12)
        for key, val in engine.scores.items():
            if key != "Composite":
                pdf.cell(200, 10, txt=safe_text(f"  {key}: {val:.2f}/100"), ln=1)
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=safe_text("Text Statistics:"), ln=1)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=safe_text(f"  Words: {engine.stats['word_count']}"), ln=1)
        pdf.cell(200, 10, txt=safe_text(f"  Sentences: {engine.stats['sentence_count']}"), ln=1)
        pdf.cell(200, 10, txt=safe_text(f"  Avg Sentence Length: {engine.stats['avg_sentence_len']:.2f}"), ln=1)
        pdf.cell(200, 10, txt=safe_text(f"  Avg Word Length: {engine.stats['avg_word_len']:.2f}"), ln=1)
        pdf.ln(3)
        # Section summaries (Medium)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=safe_text("Section Summaries (Medium):"), ln=1)
        pdf.set_font("Arial", size=10)
        for title in ['Abstract', 'Introduction', 'Methodology', 'Results', 'Conclusion']:
            summary = engine.ai_summaries.get(title, {}).get('Medium', 'N/A')
            pdf.multi_cell(0, 10, txt=safe_text(f"[{title}]"))
            pdf.multi_cell(0, 10, txt=safe_text(summary))
            pdf.ln(2)
        pdf.ln(3)
    return pdf.output(dest='S').encode('latin-1')

# --- PDF REPORT GENERATOR (single paper) ---
def create_pdf_report(filename, engine):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=safe_text(f"PaperIQ Analysis Report: {filename}"), ln=1, align='C')
    pdf.ln(10)

    # Composite score
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=safe_text(f"Final Composite Score: {engine.scores['Composite']:.2f}/100"), ln=1)
    pdf.ln(5)

    # All scores
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=safe_text("Scores:"), ln=1)
    pdf.set_font("Arial", size=12)
    for key, val in engine.scores.items():
        if key != "Composite":
            pdf.cell(200, 10, txt=safe_text(f"  {key}: {val:.2f}/100"), ln=1)
    pdf.ln(5)

    # Basic statistics
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=safe_text("Text Statistics:"), ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=safe_text(f"  Words: {engine.stats['word_count']}"), ln=1)
    pdf.cell(200, 10, txt=safe_text(f"  Sentences: {engine.stats['sentence_count']}"), ln=1)
    pdf.cell(200, 10, txt=safe_text(f"  Avg Sentence Length: {engine.stats['avg_sentence_len']:.2f}"), ln=1)
    pdf.cell(200, 10, txt=safe_text(f"  Avg Word Length: {engine.stats['avg_word_len']:.2f}"), ln=1)
    pdf.cell(200, 10, txt=safe_text(f"  Vocabulary Diversity: {engine.stats.get('vocab_diversity', 0):.2f}"), ln=1)
    pdf.cell(200, 10, txt=safe_text(f"  Complex Word Ratio: {engine.stats.get('complex_word_ratio', 0):.2f}"), ln=1)
    pdf.ln(5)

    # Domain
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=safe_text(f"Domain: {engine.domain}"), ln=1)
    pdf.ln(5)

    # Keywords
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=safe_text("Keywords:"), ln=1)
    pdf.set_font("Arial", size=12)
    if engine.present_keywords:
        pdf.cell(200, 10, txt=safe_text(f"  Present: {', '.join(engine.present_keywords)}"), ln=1)
    if engine.missing_keywords:
        pdf.cell(200, 10, txt=safe_text(f"  Missing: {', '.join(engine.missing_keywords)}"), ln=1)
    if not engine.present_keywords and not engine.missing_keywords:
        pdf.cell(200, 10, txt=safe_text("  No keywords provided."), ln=1)
    pdf.ln(5)

    # Top word frequency
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=safe_text("Top 10 Words (excluding stopwords):"), ln=1)
    pdf.set_font("Arial", size=12)
    for word, count in engine.word_freq[:10]:
        pdf.cell(200, 10, txt=safe_text(f"  {word}: {count}"), ln=1)
    pdf.ln(5)

    # Sentiment
    sentiment = engine.sentiment
    if sentiment > 0.05:
        sentiment_label = "Positive"
    elif sentiment < -0.05:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=safe_text(f"Sentiment: {sentiment:.2f} ({sentiment_label})"), ln=1)
    pdf.ln(5)

    # Section summaries (Medium length)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=safe_text("Section Summaries (Medium length):"), ln=1)
    pdf.set_font("Arial", size=10)
    for title in ['Abstract', 'Introduction', 'Methodology', 'Results', 'Conclusion']:
        summary = engine.ai_summaries.get(title, {}).get('Medium', 'N/A')
        pdf.multi_cell(0, 10, txt=safe_text(f"[{title}]"))
        pdf.multi_cell(0, 10, txt=safe_text(summary))
        pdf.ln(3)

    # Suggestions
    suggestions = engine._generate_suggestions()
    if suggestions:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=safe_text("Vocabulary Suggestions:"), ln=1)
        pdf.set_font("Arial", size=10)
        for s in suggestions:
            s_safe = safe_text(s.replace('\u2022', '-'))
            pdf.multi_cell(0, 10, txt=f"  - {s_safe}")
        pdf.ln(5)

    # Long sentences (issues)
    if engine.issues:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=safe_text("Long Sentences (>30 words):"), ln=1)
        pdf.set_font("Arial", size=10)
        for i, s in enumerate(engine.issues[:5]):
            s_safe = safe_text(s)
            pdf.multi_cell(0, 10, txt=f"  {i+1}. {s_safe}")
        if len(engine.issues) > 5:
            pdf.cell(200, 10, txt=safe_text(f"  ... and {len(engine.issues)-5} more."), ln=1)
        pdf.ln(5)

    return pdf.output(dest='S').encode('latin-1')

# --- CORE ENGINE (Improved with spaCy and cleaning) ---
class InsightEngine:
    def __init__(self):
        self.full_text = ""          # raw text from PDF
        self.clean_text = ""          # after cleaning (headers, footers, code, etc.)
        self.sections_detected = {}
        self.mandatory_map = {'Abstract': '', 'Introduction': '', 'Methodology': '', 'Results': '', 'Conclusion': ''}
        self.section_detected_flag = {k: False for k in self.mandatory_map}
        self.ai_summaries = {k: {} for k in self.mandatory_map}
        self.stats = {}
        self.word_freq = []
        self.domain = "General Research"
        self.missing_keywords = []
        self.present_keywords = []
        self.unique_user_keywords = []
        self.paper_tldr = ""
        self.paper_summaries = {'Short': '', 'Medium': '', 'Long': ''}
        # New scoring attributes
        self.scores = {}
        self.sentiment = 0.0
        self.issues = []  # long sentences (real ones)
        # spaCy processed data
        self.doc = None
        self.sentences = []   # list of sentence strings (cleaned)
        self.words = []       # list of word strings (cleaned)

    # ------------------------------
    # Cleaning methods
    # ------------------------------
    def _remove_headers_footers(self, text):
        """Remove common headers/footers like page numbers, running titles."""
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            line_strip = line.strip()
            # Skip page numbers (digits)
            if re.match(r'^\s*\d+\s*$', line_strip):
                continue
            cleaned.append(line)
        return '\n'.join(cleaned)

    def _remove_roman_numeral_lines(self, text):
        """Remove lines that consist solely of Roman numerals (page numbering)."""
        roman_numerals = {
            'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
            'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx'
        }
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            line_strip = line.strip().lower()
            if line_strip in roman_numerals:
                continue
            cleaned.append(line)
        return '\n'.join(cleaned)

    def _remove_structural_markers(self, text):
        """Remove lines that contain common structural markers (e.g., chapter headings, figure captions)."""
        structural_keywords = [
            'CHAPTER', 'FIGURE', 'TABLE', 'LIST OF', 'CERTIFICATE', 'DECLARATION',
            'ACKNOWLEDGEMENT', 'ARCHITECTURE DIAGRAM', 'USE CASE DIAGRAM', 'CLASS DIAGRAM',
            'SEQUENCE DIAGRAM', 'ACKNOWLEDGMENTS'
        ]
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            line_upper = line.strip().upper()
            # If line contains any of the structural keywords, skip it
            if any(kw in line_upper for kw in structural_keywords):
                continue
            cleaned.append(line)
        return '\n'.join(cleaned)

    def _remove_metadata_lines(self, text):
        """
        Remove lines that are likely metadata or document structure
        (title page, affiliation, declaration, etc.) based on keywords and patterns.
        """
        metadata_patterns = [
            r'\bBACHELOR\b', r'\bMASTER\b', r'\bDOCTOR\b', r'\bENGINEERING\b',
            r'\bTECHNOLOGY\b', r'\bSCIENCE\b', r'\bARTS\b', r'\bUNIVERSITY\b',
            r'\bCOLLEGE\b', r'\bINSTITUTE\b', r'\bSCHOOL\b', r'\bACADEMY\b',
            r'\bSUBMITTED BY\b', r'\bUNDER THE GUIDANCE OF\b', r'\bIN PARTIAL FULFILLMENT\b',
            r'\bFOR THE AWARD OF\b', r'\bDECLARATION\b', r'\bCERTIFICATE\b',
            r'\bACKNOWLEDGEMENT\b', r'\bACKNOWLEDGMENTS\b', r'\bDR\.', r'\bPROF\.',
            r'\bMR\.', r'\bMRS\.', r'\bMS\.', r'\bROLL NO\b', r'\bREGISTRATION NO\b',
            r'\bENROLLMENT NO\b'
        ]
        # Compile regex (case-insensitive)
        pattern = re.compile('|'.join(metadata_patterns), re.IGNORECASE)

        lines = text.split('\n')
        cleaned = []
        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                cleaned.append(line)
                continue

            # Skip lines containing metadata keywords
            if pattern.search(line_strip):
                continue

            # Skip lines that are all uppercase and longer than 20 characters (likely institution names)
            if line_strip.isupper() and len(line_strip) > 20:
                continue

            # Skip lines that consist mainly of degree abbreviations (e.g., "B.Tech", "M.Sc")
            if re.match(r'^[A-Z]\.?[A-Z]\.?', line_strip):
                continue

            cleaned.append(line)
        return '\n'.join(cleaned)

    def _remove_code_blocks(self, text):
        """Detect and remove code blocks."""
        lines = text.split('\n')
        in_code = False
        cleaned = []
        code_keywords = {'def ', 'class ', 'import ', 'from ', 'if ', 'else:', 'for ', 'while ',
                         'return ', 'print(', '=', '{', '}'}
        for line in lines:
            line_lower = line.lower()
            # Detect code start
            if any(kw in line_lower for kw in code_keywords) or line.startswith('    '):
                if not in_code:
                    in_code = True
                # skip this line
                continue
            else:
                if in_code and line.strip() == '':
                    # blank line might end code block
                    in_code = False
                    continue
                elif in_code:
                    # still inside code block, skip
                    continue
                else:
                    cleaned.append(line)
        return '\n'.join(cleaned)

    def _remove_tables(self, text):
        """Remove table-like structures (multiple spaces/tabs)."""
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            # If line contains multiple spaces or tabs, might be table row
            if re.search(r'\s{2,}', line) or '\t' in line:
                # Could be table, but also could be normal text with spaces. We'll be conservative.
                # For now, skip lines with more than one consecutive space (except indentation)
                if line.count('  ') > 2:  # heuristic
                    continue
            cleaned.append(line)
        return '\n'.join(cleaned)

    def _extract_prose(self):
        """After cleaning, use spaCy to get sentences and words, filtering out headings."""
        if not self.clean_text.strip():
            self.sentences = []
            self.words = []
            return
        self.doc = nlp(self.clean_text)
        # Get sentences, but exclude those that look like headings (short, all caps, or with digits)
        self.sentences = []
        for sent in self.doc.sents:
            text = sent.text.strip()
            # Skip if sentence is too short (< 5 words) and either all caps or contains a digit (likely heading/caption)
            words = [token.text for token in sent if not token.is_punct]
            if len(words) < 5 and (text.isupper() or re.search(r'\d', text)):
                continue
            # Also skip if it's a single word that is not a common word
            if len(words) == 1 and words[0].lower() not in ['abstract', 'introduction', 'conclusion']:
                continue
            self.sentences.append(text)
        # Get words from all sentences (excluding punctuation)
        self.words = [token.text for token in self.doc if not token.is_punct and not token.is_space]

    # ------------------------------
    # Existing methods (adapted)
    # ------------------------------
    def clean_text_func(self, text):
        # We'll keep this for backward compatibility, but we use the new cleaning pipeline.
        return text.lower().strip()  # not used in new pipeline

    def analyze_frequency(self, words):
        # Use self.words
        stop_words = set(['the','and','of','to','in','a','is','that','for','on','with','as','by','at','this','from','it','be','are','which','an','or'])
        filtered = [w for w in words if w.lower() not in stop_words and not w.isdigit() and len(w) > 2]
        return Counter(filtered).most_common(20)

    def classify_domain(self, text):
        text = text.lower()
        if any(x in text for x in ['neural network', 'deep learning', 'ai']):
            return "Artificial Intelligence"
        if any(x in text for x in ['iot', 'sensor', 'wireless']):
            return "Internet of Things"
        if any(x in text for x in ['data', 'mining', 'sentiment']):
            return "Data Science"
        return "General Research"

    def _smart_infer(self, text, keywords):
        paragraphs = text.split('\n\n')
        candidates = []
        for p in paragraphs:
            score = sum(1 for k in keywords if k in p.lower())
            if score > 0:
                candidates.append((score, p))
        candidates.sort(key=lambda x: x[0], reverse=True)
        if candidates:
            return "\n\n".join([c[1] for c in candidates[:3]])
        return "Content inferred based on document context."

    def _generate_3_summaries(self, text):
        if not text or len(text) < 50:
            return {'Short': text, 'Medium': text, 'Long': text}
        sums = {}
        if summarizer:
            try:
                input_text = text[:1500]
                s = summarizer(input_text, max_length=50, min_length=15, do_sample=False)
                sums['Short'] = s[0]['summary_text'].capitalize()
                m = summarizer(input_text, max_length=120, min_length=50, do_sample=False)
                sums['Medium'] = m[0]['summary_text'].capitalize()
                l = summarizer(input_text, max_length=250, min_length=100, do_sample=False)
                sums['Long'] = l[0]['summary_text'].capitalize()
            except:
                sums = self._heuristic_3_summaries(text)
        else:
            sums = self._heuristic_3_summaries(text)
        return sums

    def _heuristic_3_summaries(self, text):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.strip())
        return {
            'Short': " ".join(sentences[:2]) + "...",
            'Medium': " ".join(sentences[:4]) + "...",
            'Long': " ".join(sentences[:8]) + "..."
        }

    def _syllable_count(self, word):
        # Advanced syllable counter (same as before)
        word = word.lower()
        if len(word) <= 3:
            return 1
        if word.endswith('e') and not word.endswith('le') and len(word) > 3:
            test_word = word[:-1]
            if any(v in test_word for v in 'aeiouy'):
                word = test_word
        if word.endswith('ed') and len(word) > 3 and word[-3] not in 'aeiouy':
            word = word[:-2]
        if word.endswith('es') and len(word) > 3:
            word = word[:-2]
        count = 0
        vowels = "aeiouy"
        prev = False
        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev:
                count += 1
            prev = is_vowel
        return max(1, count)

    def _coherence_score(self):
        if len(self.sentences) < 2:
            return 100.0
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf = vectorizer.fit_transform(self.sentences)
            similarities = []
            for i in range(len(self.sentences)-1):
                sim = (tfidf[i] * tfidf[i+1].T).toarray()[0][0]
                similarities.append(sim)
            avg_sim = np.mean(similarities) if similarities else 0
            return max(0, min(100, avg_sim * 100))
        except:
            return 50.0

    def _reasoning_score(self):
        reasoning_indicators = [
            'because', 'since', 'if', 'then', 'implies', 'leads to', 'causes',
            'therefore', 'thus', 'hence', 'consequently', 'as a result',
            'in order to', 'so that', 'due to', 'for this reason'
        ]
        count = 0
        for sent in self.sentences:
            if any(ind in sent.lower() for ind in reasoning_indicators):
                count += 1
        indicator_ratio = count / len(self.sentences) if self.sentences else 0
        return min(85, indicator_ratio * 100)

    def _language_score(self):
        if not self.words or not self.sentences:
            return 0
        unique_words = set(w.lower() for w in self.words)
        vocab_diversity = len(unique_words) / len(self.words)
        vocab_score = vocab_diversity * 50
        avg_sent_len = len(self.words) / len(self.sentences)
        if avg_sent_len < 15:
            sent_len_score = (avg_sent_len / 15) * 25
        elif avg_sent_len > 25:
            sent_len_score = max(0, 25 - (avg_sent_len - 25) * 2)
        else:
            sent_len_score = 25
        avg_word_len = sum(len(w) for w in self.words) / len(self.words)
        word_len_score = min(avg_word_len / 10, 1) * 15
        complexity = self.clean_text.count(',') + self.clean_text.count(';') + self.clean_text.count(':')
        complexity_score = min(complexity / len(self.sentences) * 10, 10) if self.sentences else 0
        return vocab_score + sent_len_score + word_len_score + complexity_score

    def _sophistication_score(self):
        if not self.words:
            return 0
        long_words = [w for w in self.words if len(w) > 8]
        long_ratio = len(long_words) / len(self.words)
        avg_word_len = sum(len(w) for w in self.words) / len(self.words)
        word_len_component = min(avg_word_len / 10, 1) * 50
        unique_words = set(w.lower() for w in self.words)
        vocab_diversity = len(unique_words) / len(self.words)
        diversity_component = vocab_diversity * 50
        return word_len_component * 0.6 + diversity_component * 0.4

    def _readability_score(self):
        if len(self.sentences) == 0 or len(self.words) == 0:
            return 50
        total_syllables = sum(self._syllable_count(w) for w in self.words)
        score = 206.835 - 1.015 * (len(self.words) / len(self.sentences)) - 84.6 * (total_syllables / len(self.words))
        return max(0, min(100, score))

    def compute_scores(self):
        if not self.sentences:
            self.scores = {k: 0 for k in ['Language', 'Coherence', 'Reasoning',
                                          'Sophistication', 'Readability', 'Composite']}
            self.sentiment = 0
            self.issues = []
            self.stats = {
                'word_count': 0,
                'sentence_count': 0,
                'avg_sentence_len': 0,
                'avg_word_len': 0,
                'vocab_diversity': 0,
                'complex_word_ratio': 0,
                'pages': self.stats.get('pages', 0),
                'sections': len(self.sections_detected),
                'unique_words': 0,
                'time': '0 min'
            }
            return

        word_count = len(self.words)
        sentence_count = len(self.sentences)
        avg_sentence_len = word_count / sentence_count
        avg_word_len = sum(len(w) for w in self.words) / word_count
        unique_words = set(w.lower() for w in self.words)
        vocab_diversity = len(unique_words) / word_count
        complex_words = [w for w in self.words if len(w) > 6]
        complex_word_ratio = len(complex_words) / word_count

        self.stats.update({
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_len': avg_sentence_len,
            'avg_word_len': avg_word_len,
            'vocab_diversity': vocab_diversity,
            'complex_word_ratio': complex_word_ratio
        })

        coherence = self._coherence_score()
        reasoning = self._reasoning_score()
        language = self._language_score()
        sophistication = self._sophistication_score()
        readability = self._readability_score()

        language = min(100, max(0, language))
        composite = (language * 0.2 + coherence * 0.25 + reasoning * 0.2 +
                     sophistication * 0.15 + readability * 0.2)

        self.scores = {
            'Language': language,
            'Coherence': coherence,
            'Reasoning': reasoning,
            'Sophistication': sophistication,
            'Readability': readability,
            'Composite': composite
        }

        # Sentiment on cleaned text
        blob = TextBlob(self.clean_text)
        self.sentiment = blob.sentiment.polarity

        # Issues: real long sentences (>30 words)
        self.issues = [s for s in self.sentences if len(s.split()) > 30]

    def _generate_suggestions(self):
        # Use cleaned text
        weak_phrases = {
            r'\bshows?\b': 'demonstrates',
            r'\bproves?\b': 'confirms',
            r'\bgood\b': 'beneficial',
            r'\bbad\b': 'adverse',
            r'\bbig\b': 'substantial',
            r'\bvery\b': 'extremely',
            r'\bthis paper\b': 'this study',
            r'\bin this paper\b': 'in this work',
            r'\bwe do\b': 'we perform',
            r'\bwe see\b': 'we observe'
        }
        suggestions = []
        text_lower = self.clean_text.lower()
        for pattern, replacement in weak_phrases.items():
            if re.search(pattern, text_lower):
                suggestions.append(f"Consider using '{replacement}' instead of phrases matching '{pattern}'.")
        return suggestions[:5]

    def extract_research_gaps(self):
        gap_indicators = [
            r'future work', r'further research', r'limitation', r'not addressed',
            r'remains to be', r'need to investigate', r'open question', r'challenge'
        ]
        gaps = []
        for sent in self.sentences:
            sent_lower = sent.lower()
            if any(re.search(ind, sent_lower) for ind in gap_indicators):
                gaps.append(sent)
        return gaps[:5]

    def suggest_journal_conference(self):
        domain_lower = self.domain.lower()
        all_keywords = [kw.lower() for kw in self.present_keywords]
        if 'ai' in domain_lower or 'artificial intelligence' in domain_lower:
            return "Journal of Artificial Intelligence Research (JAIR)"
        elif 'machine learning' in domain_lower or any('learning' in kw for kw in all_keywords):
            return "ICML (International Conference on Machine Learning)"
        elif 'data' in domain_lower or any('data' in kw for kw in all_keywords):
            return "IEEE International Conference on Data Mining (ICDM)"
        elif 'network' in domain_lower or any('network' in kw for kw in all_keywords):
            return "IEEE/ACM Transactions on Networking"
        elif 'security' in domain_lower or any('security' in kw for kw in all_keywords):
            return "IEEE Symposium on Security and Privacy"
        else:
            return "PLOS ONE (Multidisciplinary)"

    def generate_project_idea(self):
        abstract = self.mandatory_map['Abstract'][:500]
        idea = f"**Project Idea based on this paper:**\n\n"
        idea += f"Develop a tool or system that {abstract[:200]}... "
        idea += "The project could focus on implementing the proposed methodology, "
        idea += "validating it on a new dataset, or extending it to a related domain."
        return idea

    def process_pdf(self, file_bytes, user_keywords_str=""):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_blocks = []
        for page in doc:
            blocks = page.get_text("blocks")
            blocks = [b for b in blocks if b[6] == 0]
            blocks.sort(key=lambda b: (b[1], b[0]))
            text_blocks.extend([b[4] for b in blocks])
        self.full_text = re.sub(r'(\w+)-\n\s*(\w+)', r'\1\2', "\n".join(text_blocks))
        self.full_text = re.sub(r'\n{3,}', '\n\n', self.full_text)

        # Apply cleaning pipeline
        cleaned = self.full_text
        cleaned = self._remove_headers_footers(cleaned)
        cleaned = self._remove_roman_numeral_lines(cleaned)
        cleaned = self._remove_metadata_lines(cleaned)        # <-- NEW
        cleaned = self._remove_structural_markers(cleaned)
        cleaned = self._remove_code_blocks(cleaned)
        cleaned = self._remove_tables(cleaned)
        self.clean_text = cleaned

        # Extract prose sentences and words using spaCy
        self._extract_prose()

        # Proceed with analysis
        self._analyze(user_keywords_str, len(doc))

    def process_text(self, raw_text, user_keywords_str=""):
        self.full_text = raw_text
        # Apply cleaning
        cleaned = self.full_text
        cleaned = self._remove_headers_footers(cleaned)
        cleaned = self._remove_roman_numeral_lines(cleaned)
        cleaned = self._remove_metadata_lines(cleaned)        # <-- NEW
        cleaned = self._remove_structural_markers(cleaned)
        cleaned = self._remove_code_blocks(cleaned)
        cleaned = self._remove_tables(cleaned)
        self.clean_text = cleaned
        self._extract_prose()
        self._analyze(user_keywords_str, 1)

    def _analyze(self, user_keywords_str, page_count):
        # Section detection (using full text, because headings are needed)
        header_regex = r'(?m)^(?:\d+(?:\.\d+)*\.?\s+)?([A-Z][a-zA-Z0-9\s\-\:]+)\s*$'
        lines = self.full_text.split('\n')
        current_header = None
        current_buffer = []

        def map_header(h):
            h = h.lower().strip()
            if 'abstract' in h or 'summary' in h:
                return 'Abstract'
            if 'introduction' in h or 'background' in h:
                return 'Introduction'
            if 'method' in h or 'proposed' in h or 'implementation' in h:
                return 'Methodology'
            if 'result' in h or 'experiment' in h:
                return 'Results'
            if 'conclusion' in h or 'future work' in h:
                return 'Conclusion'
            return None

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if len(line) < 60 and re.match(header_regex, line):
                if current_header:
                    text_content = "\n".join(current_buffer).strip()
                    if len(text_content) > 50:
                        self.sections_detected[current_header] = text_content
                        mapped_key = map_header(current_header)
                        if mapped_key:
                            self.mandatory_map[mapped_key] += "\n\n" + text_content
                            self.section_detected_flag[mapped_key] = True
                current_header = re.sub(r'^[\d\.\s]+', '', line).strip()
                current_buffer = []
            else:
                if current_header:
                    current_buffer.append(line)

        if current_header and current_buffer:
            text_content = "\n".join(current_buffer).strip()
            if len(text_content) > 50:
                self.sections_detected[current_header] = text_content
                mapped_key = map_header(current_header)
                if mapped_key:
                    self.mandatory_map[mapped_key] += "\n\n" + text_content
                    self.section_detected_flag[mapped_key] = True

        # Inference for missing sections
        if not self.mandatory_map['Abstract']:
            self.mandatory_map['Abstract'] = self.full_text[:1000]
        if not self.mandatory_map['Introduction']:
            self.mandatory_map['Introduction'] = self._smart_infer(self.full_text, ['introduction', 'background', 'overview'])
        if not self.mandatory_map['Methodology']:
            self.mandatory_map['Methodology'] = self._smart_infer(self.full_text, ['method', 'proposed', 'algorithm', 'system'])
        if not self.mandatory_map['Results']:
            self.mandatory_map['Results'] = self._smart_infer(self.full_text, ['result', 'performance', 'accuracy', 'table'])
        if not self.mandatory_map['Conclusion']:
            self.mandatory_map['Conclusion'] = self._smart_infer(self.full_text, ['conclusion', 'summary', 'future'])

        # Generate 3‑level summaries for sections (using raw section text)
        for key in self.mandatory_map:
            content = self.mandatory_map[key]
            if content and len(content) > 50:
                self.ai_summaries[key] = self._generate_3_summaries(content)
            else:
                self.ai_summaries[key] = {'Short': 'N/A', 'Medium': 'N/A', 'Long': 'N/A'}

        # Overall paper summary (using abstract + introduction)
        combined_source = self.mandatory_map['Abstract'] + "\n" + self.mandatory_map['Introduction']
        self.paper_summaries = self._generate_3_summaries(combined_source)
        self.paper_tldr = self.paper_summaries['Medium']

        # Keyword matching (using cleaned words)
        if user_keywords_str:
            raw_keys = [k.strip().lower() for k in user_keywords_str.split(',') if k.strip()]
            self.unique_user_keywords = sorted(list(set(raw_keys)))
            # Use cleaned text for keyword presence
            clean_lower = self.clean_text.lower()
            self.present_keywords = [k for k in self.unique_user_keywords if k in clean_lower]
            self.missing_keywords = [k for k in self.unique_user_keywords if k not in clean_lower]

        # Word frequency from cleaned words
        self.word_freq = self.analyze_frequency(self.words)
        self.domain = self.classify_domain(self.clean_text)
        self.stats.update({
            'pages': page_count,
            'sections': len(self.sections_detected),
            'words': len(self.words),
            'unique_words': len(set(self.words)),
            'time': f"{math.ceil(len(self.words)/200)} min"
        })

        # Compute scores, sentiment, issues
        self.compute_scores()

# --- VIEWS (Login, Register, Forgot Password) ---
def login_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        st.markdown("<h1 style='text-align:center;'>PaperIQ</h1>", unsafe_allow_html=True)
        email = st.text_input("Email", placeholder="Enter your email")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        if st.button("Sign In", key="login_btn"):
            user = login_user(email, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user_role = user['role']
                st.session_state.user_name = user['fullname']
                st.session_state.user_email = user['email']
                st.session_state.user_initial = user['fullname'][0].upper()
                st.rerun()
            else:
                st.error("Invalid Credentials")
        st.markdown("<br>", unsafe_allow_html=True)
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("Create Account"):
                st.session_state.page = 'register'
                st.rerun()
        with btn_c2:
            if st.button("Forgot Password?"):
                st.session_state.page = 'forgot_pwd'
                st.session_state.fp_step = 1
                st.rerun()

def forgot_password_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        st.markdown("<h1 style='text-align:center;'>Reset Password</h1>", unsafe_allow_html=True)
        st.markdown("<div style='background:#111; padding:30px; border-radius:15px; border:1px solid #333;'>", unsafe_allow_html=True)
        if st.session_state.fp_step == 1:
            email = st.text_input("Enter your registered Email")
            if st.button("Next"):
                if email:
                    sq = get_security_question(email)
                    if sq:
                        st.session_state.fp_email = email
                        st.session_state.fp_sq = sq
                        st.session_state.fp_step = 2
                        st.rerun()
                    else:
                        st.error("User does not exist")
                else:
                    st.error("Please enter your email.")
        elif st.session_state.fp_step == 2:
            st.info(f"Security Question: **{st.session_state.fp_sq}**")
            answer = st.text_input("Your Answer")
            if st.button("Verify Answer"):
                if answer:
                    if verify_security_answer(st.session_state.fp_email, answer):
                        st.session_state.fp_step = 3
                        st.rerun()
                    else:
                        st.error("Wrong security question or answer")
                else:
                    st.error("Please provide an answer.")
        elif st.session_state.fp_step == 3:
            st.success("Verified. Set new password.")
            new_pwd = st.text_input("New Password", type="password")
            confirm_pwd = st.text_input("Confirm New Password", type="password")
            if st.button("Save Password"):
                if new_pwd and confirm_pwd:
                    if new_pwd == confirm_pwd:
                        update_password(st.session_state.fp_email, new_pwd)
                        st.success("Password updated successfully!")
                        time.sleep(1.5)
                        st.session_state.page = 'login'
                        st.session_state.fp_step = 1
                        st.rerun()
                    else:
                        st.error("Passwords do not match.")
                else:
                    st.error("Fields cannot be empty.")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Back to Login"):
            st.session_state.page = 'login'
            st.session_state.fp_step = 1
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def register_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        st.markdown("<h1 style='text-align:center;'>Join PaperIQ</h1>", unsafe_allow_html=True)
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        pwd = st.text_input("Password", type="password")
        role = st.selectbox("Role", ["Student", "Lecturer"])
        sq = st.text_input("Security Question (Pet Name?)")
        sa = st.text_input("Answer")
        if st.button("Register"):
            success, msg = register_user(email, name, pwd, role, sq, sa)
            if success:
                st.success(msg)
                time.sleep(1)
                st.session_state.page = 'login'
                st.rerun()
            else:
                st.error(msg)
        if st.button("Back to Login"):
            st.session_state.page = 'login'
            st.rerun()

# --- DASHBOARD VIEW (unchanged except using new engine) ---
def dashboard_view():
    st.markdown("### Dashboard")
    st.markdown("""
        <div class='info-card'>
            <div class='info-title'>📄 PaperIQ – AI Academic Writing Analyzer</div>
            <div class='info-text'>PaperIQ is an AI-powered academic writing evaluation tool developed during Infosys Springboard Virtual Internship 6.0.<br><br>
            It analyzes research documents and provides:<br>
            • Language Quality Score<br>
            • Coherence Score<br>
            • Reasoning Strength</div>
        </div>
    """, unsafe_allow_html=True)

    col_up, col_man = st.columns(2)
    with col_up:
        st.markdown("#### Option A: Upload PDF(s)")
        uploaded_files = st.file_uploader("Drop your file(s) here", type="pdf", accept_multiple_files=True)
    with col_man:
        st.markdown("#### Option B: Manual Input")
        paper_title = st.text_input("Paper Title")
        user_keywords = st.text_input("Keywords (comma separated)")
        manual_abstract = st.text_area("Paste Abstract Text", height=100)

    if uploaded_files or manual_abstract:
        st.markdown("#### Summary Settings")
        length_option = st.select_slider(
            "Summary Detail Level",
            options=["Short", "Medium", "Long"],
            value="Medium",
            key="summary_slider"
        )
        st.session_state.summary_length = length_option

    if st.button("Analyze Paper(s)", key='analyze_btn'):
        st.session_state.analyses = {}
        st.session_state.current_filename = ""

        if manual_abstract:
            with st.spinner("Analyzing manual text..."):
                engine = InsightEngine()
                engine.process_text(manual_abstract, user_keywords)
                fname = paper_title if paper_title else "Manual Text"
                st.session_state.analyses[fname] = engine
                st.session_state.current_filename = fname

        if uploaded_files:
            progress_bar = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                with st.spinner(f"Analyzing {uploaded_file.name}..."):
                    engine = InsightEngine()
                    engine.process_pdf(uploaded_file.read(), user_keywords)
                    st.session_state.analyses[uploaded_file.name] = engine
                    run_query("INSERT INTO upload_history (user_email, file_name, page_count, word_count) VALUES (?, ?, ?, ?)",
                              (st.session_state.user_email, uploaded_file.name, engine.stats['pages'], engine.stats['words']))
                progress_bar.progress((i + 1) / len(uploaded_files))
            if not st.session_state.current_filename and uploaded_files:
                st.session_state.current_filename = uploaded_files[0].name

        if not uploaded_files and not manual_abstract:
            st.error("Please upload at least one PDF or enter text to analyze.")

    if st.session_state.analyses:
        paper_names = list(st.session_state.analyses.keys())
        selected = st.selectbox("Select paper to view", paper_names, index=paper_names.index(st.session_state.current_filename) if st.session_state.current_filename in paper_names else 0)
        st.session_state.current_filename = selected
        eng = st.session_state.analyses[selected]

        st.markdown("---")
        st.markdown(f"### Results: {st.session_state.current_filename}")

        selected_len = st.session_state.summary_length
        tldr = eng.paper_summaries.get(selected_len, eng.paper_tldr)
        st.info(f"AI Quick Summary (TL;DR): {tldr}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Composite Score", f"{eng.scores['Composite']:.2f}/100")
        with col2:
            st.metric("Language", f"{eng.scores['Language']:.2f}/100")
        with col3:
            st.metric("Coherence", f"{eng.scores['Coherence']:.2f}/100")
        with col4:
            st.metric("Reasoning", f"{eng.scores['Reasoning']:.2f}/100")

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            pdf_bytes = create_pdf_report(st.session_state.current_filename, eng)
            st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name=f"{st.session_state.current_filename}_report.pdf", mime="application/pdf")
        with col_d2:
            md_str = generate_markdown(eng, st.session_state.current_filename)
            st.download_button("📝 Download Markdown", data=md_str, file_name=f"{st.session_state.current_filename}_report.md", mime="text/markdown")
        with col_d3:
            if len(st.session_state.analyses) > 1:
                combined_pdf = generate_combined_pdf(st.session_state.analyses)
                st.download_button("📚 Combined PDF (all papers)", data=combined_pdf, file_name="all_papers_report.pdf", mime="application/pdf")

        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "Visualizations", "Section Summaries", "Issues",
            "Suggestions", "Detailed Metrics", "Sentiment",
            "Cross‑Doc Q&A", "Advanced Insights", "Similarity"
        ])

        with tab1:
            cola, colb = st.columns(2)
            with cola:
                st.markdown("#### Metric Radar")
                categories = ['Language', 'Coherence', 'Reasoning', 'Sophistication', 'Readability']
                values = [eng.scores['Language'], eng.scores['Coherence'],
                          eng.scores['Reasoning'], eng.scores['Sophistication'], eng.scores['Readability']]
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Score'))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                  showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)
            with colb:
                stats = eng.stats
                fig_bar = go.Figure(data=[
                    go.Bar(name='Avg Sentence Len', x=['Sentence'], y=[stats['avg_sentence_len']],
                           marker_color='#0068c9'),
                    go.Bar(name='Avg Word Len', x=['Word'], y=[stats['avg_word_len']],
                           marker_color='#83c9ff')
                ])
                fig_bar.update_layout(title="Average Lengths", height=350, showlegend=True)
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab2:
            st.subheader("Section Overview")
            sub_tab1, sub_tab2, sub_tab3 = st.tabs([
                "Smart Section Summaries",
                "Detected Sections",
                "Missing Sections"
            ])

            with sub_tab1:
                st.markdown("We identified the following sections and generated summaries for each (based on slider).")
                selected_len = st.session_state.summary_length
                for section in ['Abstract', 'Introduction', 'Methodology', 'Results', 'Conclusion']:
                    summaries = eng.ai_summaries.get(section, {})
                    summary = summaries.get(selected_len, 'N/A')
                    detected = eng.section_detected_flag.get(section, False)
                    status = "Detected" if detected else "Inferred"
                    with st.expander(f"{section} — {status}"):
                        st.write(summary)

            with sub_tab2:
                st.markdown("These sections were explicitly found in the document:")
                if eng.sections_detected:
                    for title, content in eng.sections_detected.items():
                        with st.expander(f"{title}"):
                            st.write(content[:2000] + ("..." if len(content) > 2000 else ""))
                else:
                    st.info("No explicit section headers were detected.")

            with sub_tab3:
                st.markdown("Important sections that were **not** detected (may have been inferred):")
                important_sections = ['Abstract', 'Introduction', 'Methodology', 'Results', 'Conclusion']
                missing = [s for s in important_sections if not eng.section_detected_flag.get(s, False)]
                if missing:
                    for section in missing:
                        st.warning(f"**{section}** – not explicitly found in the document.")
                else:
                    st.success("All important sections were detected!")

        with tab3:
            st.subheader("Long Sentences (>30 words)")
            if eng.issues:
                for i, s in enumerate(eng.issues):
                    st.warning(f"**{i+1}:** {s}")
            else:
                st.success("No long sentences found.")

        with tab4:
            st.subheader("Vocabulary Improvements")
            suggestions = eng._generate_suggestions()
            if suggestions:
                for s in suggestions:
                    st.info(s)
            else:
                st.success("Great vocabulary!")

        with tab5:
            st.subheader("Detailed Metrics")
            colm1, colm2 = st.columns(2)
            with colm1:
                st.write(f"**Words:** {eng.stats['word_count']}")
                st.write(f"**Sentences:** {eng.stats['sentence_count']}")
                st.write(f"**Avg Sentence Len:** {eng.stats['avg_sentence_len']:.2f}")
                st.write(f"**Avg Word Len:** {eng.stats['avg_word_len']:.2f}")
            with colm2:
                st.write(f"**Vocabulary Diversity:** {eng.stats.get('vocab_diversity', 0):.2f}")
                st.write(f"**Complex Word Ratio:** {eng.stats.get('complex_word_ratio', 0):.2f}")
                st.write(f"**Readability Score:** {eng.scores['Readability']:.2f}/100")

        with tab6:
            st.subheader("Sentiment Analysis")
            sentiment = eng.sentiment
            if sentiment > 0.05:
                label = "Positive"
            elif sentiment < -0.05:
                label = "Negative"
            else:
                label = "Neutral"
            st.metric("Sentiment Polarity", f"{sentiment:.2f} ({label})")

        with tab7:
            st.subheader("Cross‑Document Q&A")
            st.markdown("Ask a question and get answers from all uploaded papers.")
            question = st.text_input("Your question")
            search_clicked = st.button("Search")
            if search_clicked and question:
                all_chunks = []
                metadata = []
                for fname, engine in st.session_state.analyses.items():
                    # Use cleaned sentences for chunks
                    sentences = engine.sentences
                    chunk_size = 3
                    overlap = 1
                    for i in range(0, len(sentences), chunk_size - overlap):
                        chunk = " ".join(sentences[i:i+chunk_size])
                        if chunk.strip():
                            all_chunks.append(chunk)
                            metadata.append(fname)
                if not all_chunks:
                    st.warning("No text available.")
                else:
                    vectorizer = TfidfVectorizer(stop_words='english')
                    try:
                        chunk_vectors = vectorizer.fit_transform(all_chunks)
                        query_vec = vectorizer.transform([question])
                        similarities = cosine_similarity(query_vec, chunk_vectors).flatten()
                        top_indices = similarities.argsort()[-5:][::-1]
                        st.markdown("**Top relevant excerpts:**")
                        found = False
                        for idx in top_indices:
                            if similarities[idx] > 0.1:
                                st.info(f"**{metadata[idx]}** (score: {similarities[idx]:.2f})\n\n{all_chunks[idx]}")
                                found = True
                        if not found:
                            st.markdown("*No highly relevant sentences found.*")
                    except Exception as e:
                        st.error(f"Error in search: {e}")

        with tab8:
            st.subheader("Advanced Insights")
            st.markdown("#### Research Gap Detection")
            gaps = eng.extract_research_gaps()
            if gaps:
                for g in gaps:
                    st.write(f"- {g}")
            else:
                st.info("No explicit research gaps found in the text.")

            st.markdown("#### Journal / Conference Recommendation")
            rec = eng.suggest_journal_conference()
            st.success(f"**Recommended venue:** {rec}")

            st.markdown("#### Project Idea Generator")
            idea = eng.generate_project_idea()
            st.info(idea)

        with tab9:
            st.subheader("Paper Similarity")
            if len(st.session_state.analyses) < 2:
                st.warning("Upload at least two papers to compute similarity.")
            else:
                names = list(st.session_state.analyses.keys())
                texts = [st.session_state.analyses[n].clean_text for n in names]  # use cleaned text
                vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
                try:
                    tfidf = vectorizer.fit_transform(texts)
                    sim_matrix = cosine_similarity(tfidf)
                    fig = px.imshow(sim_matrix,
                                    x=names,
                                    y=names,
                                    color_continuous_scale='Blues',
                                    title="Cosine Similarity between Papers")
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not compute similarity: {e}")

        if st.button("Save to Library"):
            run_query("INSERT INTO saved_papers (user_email, file_name, summary_abstract) VALUES (?, ?, ?)",
                      (st.session_state.user_email, st.session_state.current_filename,
                       eng.mandatory_map['Abstract'][:200]))
            st.success("Paper saved to your library!")

# --- Other views (Saved, History, Profile) ---
def saved_view():
    st.markdown("### Saved Papers")
    data = run_query("SELECT * FROM saved_papers WHERE user_email=?", (st.session_state.user_email,), fetch_all=True)
    if data:
        for d in data:
            st.markdown(f"<div class='info-card'><b>{d['file_name']}</b><br><small>{d['saved_time']}</small></div>", unsafe_allow_html=True)
    else:
        st.info("No saved items.")

def history_view():
    st.markdown("### Upload History")
    data = run_query("SELECT * FROM upload_history WHERE user_email=? ORDER BY upload_time DESC", (st.session_state.user_email,), fetch_all=True)
    if data:
        for d in data:
            st.markdown(f"<div class='info-card'><b>{d['file_name']}</b><br><small>{d['upload_time']}</small></div>", unsafe_allow_html=True)
    else:
        st.info("No history.")

def profile_view():
    st.markdown("### User Profile")
    st.markdown(f"<div class='info-card'><h2>{st.session_state.user_name}</h2><p>{st.session_state.user_email}</p><p>Role: {st.session_state.user_role}</p></div>", unsafe_allow_html=True)

# --- MAIN APP LOGIC ---
if not st.session_state.logged_in:
    if st.session_state.page == 'login':
        login_page()
    elif st.session_state.page == 'register':
        register_page()
    elif st.session_state.page == 'forgot_pwd':
        forgot_password_page()
else:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("Dashboard", key="nav_dash"):
            st.session_state.menu_selection = 'Dashboard'
            st.rerun()
    with c2:
        if st.button("Saved Papers", key="nav_saved"):
            st.session_state.menu_selection = 'Saved'
            st.rerun()
    with c3:
        if st.button("Upload History", key="nav_hist"):
            st.session_state.menu_selection = 'History'
            st.rerun()
    with c4:
        if st.button("Logout", key='logout_btn'):
            st.session_state.logged_in = False
            st.rerun()
    with c5:
        if st.button(st.session_state.user_initial, key='profile_circle'):
            st.session_state.menu_selection = 'Profile'
            st.rerun()

    st.markdown("---")

    if st.session_state.menu_selection == 'Dashboard':
        dashboard_view()
    elif st.session_state.menu_selection == 'Saved':
        saved_view()
    elif st.session_state.menu_selection == 'History':
        history_view()
    elif st.session_state.menu_selection == 'Profile':
        profile_view()