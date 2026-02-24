import fitz  # PyMuPDF
import re
import spacy

# Load English tokenizer, tagger, parser and NER
# Note: User must run `python -m spacy download en_core_web_sm`
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

def extract_text_with_metadata(pdf_file):
    """
    Extracts text from a PDF file object (Streamlit upload).
    Returns a dictionary of sections.
    """
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    full_text = ""
    
    # Simple extraction for Milestone 1
    for page in doc:
        full_text += page.get_text() + "\n"
    
    doc.close()
    return full_text

def preprocess_text(text):
    """
    Cleans the text using Regex and Spacy.
    """
    # 1. Basic Cleaning
    # Remove multiple newlines and extra spaces
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    # 2. NLP Cleaning (Optional for M1, but good for structure)
    if nlp:
        doc = nlp(text[:100000]) # Limit char count for speed in M1
        # Example: Filter out stop words or punctuation if needed later
        # for now, we just return the cleaned string
        return text.strip()
    return text.strip()

def identify_sections(text):
    """
    Uses Regex to find standard research paper sections.
    """
    sections = {
        "Abstract": "",
        "Introduction": "",
        "Methodology": "",
        "Results": "",
        "Conclusion": ""
    }
    # Common headers in research papers (Case insensitive)
    # This captures text between "Abstract" and "Introduction", etc.
    # Note: This is a simplified regex approach for Milestone 1. 
    # Real-world PDFs are messy, so we use fuzzy logic in later milestones.
    lower_text = text.lower()
    # Helper to find start indices
    def find_start(keyword):
        match = re.search(r'\b' + re.escape(keyword) + r'\b', lower_text)
        return match.start() if match else -1
    idx_abstract = find_start("abstract")
    idx_intro = find_start("introduction")
    idx_method = find_start("methodology") 
    if idx_method == -1: idx_method = find_start("methods") # Fallback
    idx_results = find_start("results")
    idx_conc = find_start("conclusion")
    idx_ref = find_start("references")
    # Slice text based on indices found
    # Logic: If we found Abstract and Intro, text between them is the Abstract content
    if idx_abstract != -1 and idx_intro != -1:
        sections["Abstract"] = text[idx_abstract:idx_intro]
    elif idx_abstract != -1:
        # If no intro found, take next 500 chars
        sections["Abstract"] = text[idx_abstract:idx_abstract+1000]
    if idx_intro != -1 and idx_method != -1:
        sections["Introduction"] = text[idx_intro:idx_method]
    elif idx_intro != -1:
         sections["Introduction"] = text[idx_intro:idx_intro+1500]
    if idx_method != -1 and idx_results != -1:
        sections["Methodology"] = text[idx_method:idx_results]
    if idx_results != -1 and idx_conc != -1:
        sections["Results"] = text[idx_results:idx_conc]
    if idx_conc != -1 and idx_ref != -1:
        sections["Conclusion"] = text[idx_conc:idx_ref]
    elif idx_conc != -1:
        sections["Conclusion"] = text[idx_conc:]
    return sections