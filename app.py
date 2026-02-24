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
from fpdf import FPDF
from sklearn.feature_extraction.text import TfidfVectorizer

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PaperIQ",
    page_icon="■",
    layout="wide",
)

# --- THEME MANAGEMENT ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'  # default theme

def toggle_theme():
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'

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
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
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

# --- CSS STYLING (Themed, with key-based button styles) ---
def get_css(theme):
    if theme == 'dark':
        bg_color = "#050505"
        card_bg = "#111111"
        text_color = "#FFFFFF"
        text_dim = "#A0A0A0"
        border = "#333333"
        button_bg = "#1E1E1E"
        primary_blue = "#0056b3"
        input_bg = "#0F0F0F"
        expander_bg = "#1A1A1A"
        expander_content_bg = "#0A0A0A"
        stat_box_bg = "#161616"
        logout_bg = "#330000"
        logout_hover = "#FF0000"
        logout_text = "#FF4444"
        theme_button_bg = "#2A2A2A"
        theme_button_text = "#FFFFFF"
        file_uploader_bg = "#0F0F0F"
        file_uploader_text = "#FFFFFF"
        file_uploader_border = "#333333"
    else:  # light theme
        bg_color = "#F5F5F5"
        card_bg = "#FFFFFF"
        text_color = "#000000"
        text_dim = "#333333"
        border = "#CCCCCC"
        button_bg = "#E0E0E0"
        primary_blue = "#007BFF"
        input_bg = "#FFFFFF"
        expander_bg = "#F0F0F0"
        expander_content_bg = "#FAFAFA"
        stat_box_bg = "#F0F0F0"
        logout_bg = "#FFCCCC"
        logout_hover = "#FF9999"
        logout_text = "#990000"
        theme_button_bg = "#DDDDDD"
        theme_button_text = "#000000"
        file_uploader_bg = "#FFFFFF"
        file_uploader_text = "#000000"
        file_uploader_border = "#CCCCCC"

    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

        .stApp {{
            background-color: {bg_color};
            font-family: 'Inter', sans-serif;
            color: {text_color};
        }}

        h1, h2, h3, h4 {{ color: {text_color} !important; font-weight: 800; }}
        p, label, span, div {{ color: {text_dim}; }}

        /* --- Navigation buttons styled by key --- */
        /* Dashboard, Saved Papers, Upload History */
        div.stButton > button[key="nav_dash"],
        div.stButton > button[key="nav_saved"],
        div.stButton > button[key="nav_hist"] {{
            background-color: {button_bg};
            color: {text_color};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
            width: 100%;
        }}
        div.stButton > button[key="nav_dash"]:hover,
        div.stButton > button[key="nav_saved"]:hover,
        div.stButton > button[key="nav_hist"]:hover {{
            background-color: {border};
            border-color: {text_color};
        }}

        /* Theme toggle button */
        div.stButton > button[key="theme_toggle"] {{
            background-color: {theme_button_bg};
            color: {theme_button_text};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
            width: 100%;
        }}
        div.stButton > button[key="theme_toggle"]:hover {{
            background-color: {border};
        }}

        /* Logout button */
        div.stButton > button[key="logout_btn"] {{
            background-color: {logout_bg} !important;
            color: {logout_text} !important;
            border: 1px solid {logout_text} !important;
            border-radius: 8px;
            font-weight: 600;
            width: 100%;
        }}
        div.stButton > button[key="logout_btn"]:hover {{
            background-color: {logout_hover} !important;
        }}

        /* Profile circle button */
        div.stButton > button[key="profile_circle"] {{
            background-color: {primary_blue};
            color: white;
            border-radius: 50%;
            width: 45px;
            height: 45px;
            padding: 0;
            font-size: 18px;
            font-weight: 800;
            border: 2px solid {text_color};
            display: flex;
            align-items: center;
            justify-content: center;
            margin-left: auto;
            margin-right: auto;
        }}
        div.stButton > button[key="profile_circle"]:hover {{
            background-color: #004494;
            transform: scale(1.1);
        }}

        /* Analyze button */
        div.stButton > button[key="analyze_btn"] {{
            background: linear-gradient(90deg, {primary_blue} 0%, #004494 100%);
            color: white;
            border: none;
            padding: 15px;
            font-size: 1.1rem;
            border-radius: 8px;
            margin-top: 20px;
            width: 100%;
        }}
        div.stButton > button[key="analyze_btn"]:hover {{
            opacity: 0.9;
            box-shadow: 0 0 15px rgba(0,86,179,0.5);
        }}

        /* File uploader */
        .stFileUploader {{
            background-color: {file_uploader_bg};
            color: {file_uploader_text};
            border: 1px dashed {file_uploader_border};
            border-radius: 8px;
            padding: 20px;
        }}
        .stFileUploader label {{
            color: {file_uploader_text} !important;
        }}
        .stFileUploader small {{
            color: {text_dim} !important;
        }}
        .stFileUploader button {{
            background-color: {primary_blue};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 5px 10px;
        }}

        /* Info cards */
        .info-card {{
            background-color: {card_bg};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
        }}
        .info-title {{ color: {text_color}; font-size: 1.2rem; font-weight: 700; margin-bottom: 10px; }}
        .info-text {{ color: {text_dim}; font-size: 0.95rem; line-height: 1.6; }}

        /* Input fields */
        input[type="text"], input[type="password"], textarea {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
            border: 1px solid {border} !important;
            border-radius: 8px !important;
        }}

        /* Expanders */
        .streamlit-expanderHeader {{
            background-color: {expander_bg} !important;
            border: 1px solid {border} !important;
            color: {text_color} !important;
            border-radius: 8px !important;
        }}
        div[data-testid="stExpander"] details > div {{
            background-color: {expander_content_bg};
            border: 1px solid {border};
            border-top: none;
            padding: 20px;
        }}

        /* Stat boxes */
        .stat-box {{ background: {stat_box_bg}; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid {border}; }}
        .stat-val {{ font-size: 1.5rem; font-weight: 800; color: {text_color}; }}
        .stat-lbl {{ font-size: 0.8rem; text-transform: uppercase; color: {text_dim}; }}

        /* Metrics (st.metric) */
        div[data-testid="stMetricValue"] {{
            color: {text_color} !important;
        }}
        div[data-testid="stMetricLabel"] {{
            color: {text_dim} !important;
        }}

        /* Tabs */
        button[data-baseweb="tab"] {{
            color: {text_color} !important;
        }}
        div[data-testid="stTabs"] {{
            background-color: {card_bg};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 10px;
        }}

        /* Select slider */
        div[data-testid="stSlider"] {{
            color: {text_color};
        }}

        #MainMenu, footer, header {{ visibility: hidden; }}
    </style>
    """

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

# --- ENHANCED PDF REPORT GENERATOR ---
def create_pdf_report(filename, engine):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"PaperIQ Analysis Report: {filename}", ln=1, align='C')
    pdf.ln(10)

    # Composite score
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Final Composite Score: {engine.scores['Composite']:.2f}/100", ln=1)
    pdf.ln(5)

    # All scores
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Scores:", ln=1)
    pdf.set_font("Arial", size=12)
    for key, val in engine.scores.items():
        if key != "Composite":
            pdf.cell(200, 10, txt=f"  {key}: {val:.2f}/100", ln=1)
    pdf.ln(5)

    # Basic statistics
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Text Statistics:", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"  Words: {engine.stats['word_count']}", ln=1)
    pdf.cell(200, 10, txt=f"  Sentences: {engine.stats['sentence_count']}", ln=1)
    pdf.cell(200, 10, txt=f"  Avg Sentence Length: {engine.stats['avg_sentence_len']:.2f}", ln=1)
    pdf.cell(200, 10, txt=f"  Avg Word Length: {engine.stats['avg_word_len']:.2f}", ln=1)
    pdf.cell(200, 10, txt=f"  Vocabulary Diversity: {engine.stats.get('vocab_diversity', 0):.2f}", ln=1)
    pdf.cell(200, 10, txt=f"  Complex Word Ratio: {engine.stats.get('complex_word_ratio', 0):.2f}", ln=1)
    pdf.ln(5)

    # Domain
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=f"Domain: {engine.domain}", ln=1)
    pdf.ln(5)

    # Keywords
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Keywords:", ln=1)
    pdf.set_font("Arial", size=12)
    if engine.present_keywords:
        pdf.cell(200, 10, txt=f"  Present: {', '.join(engine.present_keywords)}", ln=1)
    if engine.missing_keywords:
        pdf.cell(200, 10, txt=f"  Missing: {', '.join(engine.missing_keywords)}", ln=1)
    if not engine.present_keywords and not engine.missing_keywords:
        pdf.cell(200, 10, txt="  No keywords provided.", ln=1)
    pdf.ln(5)

    # Top word frequency
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Top 10 Words (excluding stopwords):", ln=1)
    pdf.set_font("Arial", size=12)
    for word, count in engine.word_freq[:10]:
        pdf.cell(200, 10, txt=f"  {word}: {count}", ln=1)
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
    pdf.cell(200, 10, txt=f"Sentiment: {sentiment:.2f} ({sentiment_label})", ln=1)
    pdf.ln(5)

    # Section summaries (Medium length)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Section Summaries (Medium length):", ln=1)
    pdf.set_font("Arial", size=10)
    for title in ['Abstract', 'Introduction', 'Methodology', 'Results', 'Conclusion']:
        summary = engine.ai_summaries.get(title, {}).get('Medium', 'N/A')
        safe_title = title.encode('latin-1', 'replace').decode('latin-1')
        safe_summary = summary.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, txt=f"[{safe_title}]")
        pdf.multi_cell(0, 10, txt=safe_summary)
        pdf.ln(3)

    # Suggestions
    suggestions = engine._generate_suggestions()
    if suggestions:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Vocabulary Suggestions:", ln=1)
        pdf.set_font("Arial", size=10)
        for s in suggestions:
            # Replace bullet with hyphen for safety
            s_safe = s.replace('\u2022', '-').encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 10, txt=f"  - {s_safe}")
        pdf.ln(5)

    # Long sentences (issues)
    if engine.issues:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Long Sentences (>30 words):", ln=1)
        pdf.set_font("Arial", size=10)
        for i, s in enumerate(engine.issues[:5]):  # limit to first 5
            s_safe = s.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 10, txt=f"  {i+1}. {s_safe}")
        if len(engine.issues) > 5:
            pdf.cell(200, 10, txt=f"  ... and {len(engine.issues)-5} more.", ln=1)
        pdf.ln(5)

    return pdf.output(dest='S').encode('latin-1')

# --- CORE ENGINE (Improved scoring) ---
class InsightEngine:
    def __init__(self):
        self.full_text = ""
        self.clean_text_content = ""
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
        self.issues = []

    def clean_text_func(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def analyze_frequency(self, text):
        words = text.split()
        stop_words = set(['the','and','of','to','in','a','is','that','for','on','with','as','by','at','this','from','it','be','are','which','an','or'])
        filtered_words = [w for w in words if w not in stop_words and not w.isdigit() and len(w) > 2]
        return Counter(filtered_words).most_common(20)

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
        """Simple syllable counter based on vowel groups."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        prev_is_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        if word.endswith("e"):
            count -= 1
        if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
            count += 1
        return max(1, count)

    def _coherence_score(self, sentences):
        """Compute coherence as average cosine similarity between consecutive sentences using TF-IDF."""
        if len(sentences) < 2:
            return 100.0
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf_matrix = vectorizer.fit_transform([s.lower() for s in sentences])
            similarities = []
            for i in range(len(sentences) - 1):
                sim = (tfidf_matrix[i] * tfidf_matrix[i+1].T).toarray()[0][0]
                similarities.append(sim)
            avg_sim = sum(similarities) / len(similarities) if similarities else 0
            return max(0, min(100, avg_sim * 100))
        except Exception:
            # Fallback if sklearn fails (e.g., empty vocabulary)
            return 50.0

    def _reasoning_score(self, sentences, words):
        """Detect reasoning using indicator patterns and sentence position."""
        reasoning_indicators = [
            'because', 'since', 'if', 'then', 'implies', 'leads to', 'causes',
            'therefore', 'thus', 'hence', 'consequently', 'as a result',
            'in order to', 'so that', 'due to', 'for this reason'
        ]
        # Count sentences containing any indicator
        indicator_sentences = 0
        for sent in sentences:
            if any(ind in sent.lower() for ind in reasoning_indicators):
                indicator_sentences += 1

        # Also look for cause-effect patterns (e.g., "X causes Y")
        pattern_score = 0
        cause_effect_patterns = [
            r'\b\w+\s+causes?\s+\w+',
            r'\b\w+\s+leads?\s+to\s+\w+',
            r'\bif\s+.+\s+then\s+.+',
            r'\bdue\s+to\s+.+'
        ]
        for pattern in cause_effect_patterns:
            if re.search(pattern, self.full_text.lower()):
                pattern_score += 20

        # Weighted score, capped at 85
        indicator_ratio = indicator_sentences / len(sentences) if sentences else 0
        raw_score = indicator_ratio * 60 + min(pattern_score, 40)
        return min(85, raw_score)

    def _language_score(self, words, sentences, word_count, sentence_count):
        """Combine vocabulary diversity, sentence length, word length, and complexity."""
        if word_count == 0 or sentence_count == 0:
            return 0

        # Vocabulary diversity
        unique_words = set(w.lower() for w in words)
        vocab_diversity = len(unique_words) / word_count
        vocab_score = vocab_diversity * 50  # max 50 (if all words unique)

        # Sentence length (ideal 15-25 words)
        avg_sent_len = word_count / sentence_count
        if avg_sent_len < 15:
            sent_len_score = (avg_sent_len / 15) * 25
        elif avg_sent_len > 25:
            sent_len_score = max(0, 25 - (avg_sent_len - 25) * 2)
        else:
            sent_len_score = 25

        # Word length (longer words often more sophisticated)
        avg_word_len = sum(len(w) for w in words) / word_count
        word_len_score = min(avg_word_len / 10, 1) * 15

        # Sentence complexity (use punctuation as proxy)
        complexity = self.full_text.count(',') + self.full_text.count(';') + self.full_text.count(':')
        complexity_score = min(complexity / sentence_count * 10, 10)

        return vocab_score + sent_len_score + word_len_score + complexity_score

    def _sophistication_score(self, words, word_count):
        """Lexical sophistication based on word length and diversity."""
        if word_count == 0:
            return 0
        # Long words (>8 chars) are often technical
        long_words = [w for w in words if len(w) > 8]
        long_ratio = len(long_words) / word_count

        # Average word length
        avg_word_len = sum(len(w) for w in words) / word_count
        word_len_component = min(avg_word_len / 10, 1) * 50

        # Vocabulary diversity again
        unique_words = set(w.lower() for w in words)
        vocab_diversity = len(unique_words) / word_count
        diversity_component = vocab_diversity * 50

        return word_len_component * 0.6 + diversity_component * 0.4

    def _readability_score(self, words, sentences, word_count, sentence_count):
        """Flesch Reading Ease with accurate syllable counting."""
        if sentence_count == 0 or word_count == 0:
            return 50
        total_syllables = sum(self._syllable_count(w) for w in words)
        score = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (total_syllables / word_count)
        return max(0, min(100, score))

    def compute_scores(self):
        blob = TextBlob(self.full_text)
        sentences = [str(s) for s in blob.sentences]
        words = blob.words
        word_count = len(words)
        sentence_count = len(sentences)

        if word_count == 0 or sentence_count == 0:
            self.scores = {k: 0 for k in ['Language', 'Coherence', 'Reasoning',
                                          'Sophistication', 'Readability', 'Composite']}
            self.sentiment = 0
            self.issues = []
            return

        # Basic stats (store for UI)
        avg_sentence_len = word_count / sentence_count
        avg_word_len = sum(len(w) for w in words) / word_count
        unique_words = set(w.lower() for w in words)
        vocab_diversity = len(unique_words) / word_count
        complex_words = [w for w in words if len(w) > 6]
        complex_word_ratio = len(complex_words) / word_count

        self.stats.update({
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_len': avg_sentence_len,
            'avg_word_len': avg_word_len,
            'vocab_diversity': vocab_diversity,
            'complex_word_ratio': complex_word_ratio
        })

        # Compute individual scores
        coherence = self._coherence_score(sentences)
        reasoning = self._reasoning_score(sentences, words)
        language = self._language_score(words, sentences, word_count, sentence_count)
        sophistication = self._sophistication_score(words, word_count)
        readability = self._readability_score(words, sentences, word_count, sentence_count)

        # Normalize language to 0-100 (it may already be close, but ensure)
        language = min(100, max(0, language))

        # Composite (weighted)
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

        # Sentiment
        self.sentiment = blob.sentiment.polarity

        # Issues (long sentences)
        self.issues = [s for s in sentences if len(s.split()) > 30]

    def _generate_suggestions(self):
        """Return a list of context-aware suggestions."""
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
        for pattern, replacement in weak_phrases.items():
            if re.search(pattern, self.full_text.lower()):
                suggestions.append(f"Consider using '{replacement}' instead of phrases matching '{pattern}'.")
        return suggestions[:5]  # limit to top 5

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
        self._analyze(user_keywords_str, len(doc))

    def process_text(self, raw_text, user_keywords_str=""):
        self.full_text = raw_text
        self._analyze(user_keywords_str, 1)

    def _analyze(self, user_keywords_str, page_count):
        self.clean_text_content = self.clean_text_func(self.full_text)
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

        # Inference for missing sections (only if not already detected)
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

        # Generate 3‑level summaries for sections
        for key in self.mandatory_map:
            content = self.mandatory_map[key]
            if content and len(content) > 50:
                self.ai_summaries[key] = self._generate_3_summaries(content)
            else:
                self.ai_summaries[key] = {'Short': 'N/A', 'Medium': 'N/A', 'Long': 'N/A'}

        # Generate overall paper summaries (Abstract + Introduction)
        combined_source = self.mandatory_map['Abstract'] + "\n" + self.mandatory_map['Introduction']
        self.paper_summaries = self._generate_3_summaries(combined_source)
        self.paper_tldr = self.paper_summaries['Medium']

        # Keyword matching
        words_list = self.clean_text_content.split()
        if user_keywords_str:
            raw_keys = [k.strip().lower() for k in user_keywords_str.split(',') if k.strip()]
            self.unique_user_keywords = sorted(list(set(raw_keys)))
            self.present_keywords = [k for k in self.unique_user_keywords if k in self.clean_text_content]
            self.missing_keywords = [k for k in self.unique_user_keywords if k not in self.clean_text_content]

        self.word_freq = self.analyze_frequency(self.clean_text_content)
        self.domain = self.classify_domain(self.clean_text_content)
        self.stats.update({
            'pages': page_count,
            'sections': len(self.sections_detected),
            'words': len(words_list),
            'unique_words': len(set(words_list)),
            'time': f"{math.ceil(len(words_list)/200)} min"
        })

        # Compute scores, sentiment, issues
        self.compute_scores()

    def generate_report(self):
        return f"PaperIQ Report\nDomain: {self.domain}\nSummary: {self.mandatory_map['Abstract'][:500]}"

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

# --- DASHBOARD VIEW ---
def dashboard_view():
    st.markdown("### Dashboard")
    # Updated Info Card as requested
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

    # Upload area
    col_up, col_man = st.columns(2)
    with col_up:
        st.markdown("#### Option A: Upload PDF")
        uploaded_file = st.file_uploader("Drop your file here", type="pdf")
    with col_man:
        st.markdown("#### Option B: Manual Input")
        paper_title = st.text_input("Paper Title")
        user_keywords = st.text_input("Keywords (comma separated)")
        manual_abstract = st.text_area("Paste Abstract Text", height=100)

    # Summary settings slider
    if uploaded_file or manual_abstract:
        st.markdown("#### Summary Settings")
        length_option = st.select_slider(
            "Summary Detail Level",
            options=["Short", "Medium", "Long"],
            value="Medium",
            key="summary_slider"
        )
        st.session_state.summary_length = length_option

    if st.button("Analyze Paper", key='analyze_btn'):
        engine = InsightEngine()
        if uploaded_file:
            with st.spinner("AI is reading & summarizing..."):
                engine.process_pdf(uploaded_file.read(), user_keywords)
                st.session_state.current_analysis = engine
                st.session_state.current_filename = uploaded_file.name
                run_query("INSERT INTO upload_history (user_email, file_name, page_count, word_count) VALUES (?, ?, ?, ?)",
                          (st.session_state.user_email, uploaded_file.name, engine.stats['pages'], engine.stats['words']))
        elif manual_abstract:
            with st.spinner("AI is reading & summarizing..."):
                engine.process_text(manual_abstract, user_keywords)
                st.session_state.current_analysis = engine
                st.session_state.current_filename = paper_title if paper_title else "Manual Text"
        else:
            st.error("Please Upload a PDF or Enter Text to analyze.")

    # Display results if available
    if st.session_state.current_analysis:
        eng = st.session_state.current_analysis
        st.markdown("---")
        st.markdown(f"### Results: {st.session_state.current_filename}")

        # Quick TL;DR - dynamic based on slider selection
        selected_len = st.session_state.summary_length
        tldr = eng.paper_summaries.get(selected_len, eng.paper_tldr)
        st.info(f"AI Quick Summary (TL;DR): {tldr}")

        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Composite Score", f"{eng.scores['Composite']:.2f}/100")
        with col2:
            st.metric("Language", f"{eng.scores['Language']:.2f}/100")
        with col3:
            st.metric("Coherence", f"{eng.scores['Coherence']:.2f}/100")
        with col4:
            st.metric("Reasoning", f"{eng.scores['Reasoning']:.2f}/100")

        # Download PDF button (using enhanced report)
        pdf_bytes = create_pdf_report(st.session_state.current_filename, eng)
        st.download_button("Download PDF Report", data=pdf_bytes, file_name="PaperIQ_Report.pdf", mime="application/pdf")

        st.markdown("---")

        # Tabs (matches the 6 tabs from user's provided code)
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Visualizations", "Section Summaries", "Issues",
            "Suggestions", "Detailed Metrics", "Sentiment"
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
                st.plotly_chart(fig, width='stretch')
            with colb:
                st.markdown("#### Text Statistics")
                stats = eng.stats
                fig_bar = go.Figure(data=[
                    go.Bar(name='Avg Sentence Len', x=['Sentence'], y=[stats['avg_sentence_len']],
                           marker_color='#0068c9'),
                    go.Bar(name='Avg Word Len', x=['Word'], y=[stats['avg_word_len']],
                           marker_color='#83c9ff')
                ])
                fig_bar.update_layout(title="Average Lengths", height=350, showlegend=True)
                st.plotly_chart(fig_bar, width='stretch')

        # --- Section Summaries tab with three sub-tabs ---
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
                    # Gets the short, medium, or long summary dynamically based on slider!
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
                sentiment_label = "Positive"
            elif sentiment < -0.05:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
            st.metric("Sentiment Polarity", f"{sentiment:.2f} ({sentiment_label})")

        # Save to Library (optional)
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
    # Apply theme CSS
    st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)

    # Top navigation with equal spacing (6 columns)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
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