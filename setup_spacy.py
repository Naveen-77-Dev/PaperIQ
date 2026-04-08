#!/usr/bin/env python3
"""Initialize spaCy models for Streamlit deployment"""
import subprocess
import sys

try:
    import spacy
    spacy.load("en_core_web_sm")
except (OSError, ImportError):
    print("Installing spaCy English model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    except Exception as e:
        print(f"Warning: Could not install spaCy model: {e}")
        print("The app will work with a lightweight fallback model.")
