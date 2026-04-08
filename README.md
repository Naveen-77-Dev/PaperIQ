Features

**Core Analysis**

Language Quality Score: Evaluates vocabulary diversity, sentence structure, and word choice
Coherence Score: Measures logical flow using TF-IDF similarity between consecutive sentences
Reasoning Strength: Detects logical connectors and cause-effect patterns
Sophistication Score: Assesses lexical complexity and academic tone
Readability Score: Calculates Flesch Reading Ease with accurate syllable counting
Composite Score: Weighted combination of all metrics (0-100)

**Advanced Features**

AI-Powered Section Summaries: Auto-detects Abstract, Introduction, Methodology, Results, Conclusion
Multi-level Summaries: Short, Medium, and Long summaries for each section
Sentiment Analysis: Polarity detection using TextBlob
Keyword Matching: Track present/missing keywords in your paper
Domain Classification: Auto-identifies research domain (AI, IoT, Data Science, etc.)
Long Sentence Detection: Flags complex sentences (>30 words)
Vocabulary Suggestions: AI-driven improvements for weak phrases
Research Gap Detection: Identifies limitations and future work indicators
Cross-Document Q&A: Search across multiple papers with TF-IDF similarity
Paper Similarity Analysis: Compute similarity between uploaded papers

**Export & Reports**

PDF Reports: Comprehensive single-paper or combined reports
Markdown Export: Download analysis as markdown for documentation
Combined PDF: Generate reports for all uploaded papers at once

**User Management**

User registration and login with password hashing
Role-based access (Student/Lecturer)
Security questions for password recovery
Upload history tracking
Paper library for saving documents

**Installation**

**Requirements**

Python 3.8+
Streamlit
PyMuPDF (fitz)
scikit-learn
TextBlob
Plotly
FPDF
transformers (for AI summaries)

**Setup**

Creating and activating virtual environment

Windows
python -m venv venv
venv\Scripts\Activate

macOS / Linux
python3 -m venv venv
source venv/bin/Activate

Install required libraries
pip install -r requirements.txt
Install spaCy model
python -m spacy download en_core_web_sm
Run the project
streamlit run app.py
