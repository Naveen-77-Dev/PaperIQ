#!/usr/bin/env python3
"""Setup script to download spaCy model for Streamlit deployment"""
import os
import sys
import subprocess

def download_spacy_model():
    """Download spaCy English model"""
    try:
        import spacy
        try:
            # Check if model is already installed
            spacy.load("en_core_web_sm")
            print("✓ spaCy model 'en_core_web_sm' is already installed")
            return True
        except OSError:
            print("Downloading spaCy English model (en_core_web_sm)...")
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("✓ spaCy model downloaded successfully")
            return True
    except Exception as e:
        print(f"⚠ Could not download spaCy model: {e}")
        print("The app will use a lightweight fallback model.")
        return False

if __name__ == "__main__":
    download_spacy_model()

