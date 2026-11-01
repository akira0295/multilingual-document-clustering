import pymupdf  # PyMuPDF
import pandas as pd
from pathlib import Path
import re
import unicodedata
from preprocess import clean_text, detect_language

def extract_text(path):
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".txt":
        return path.read_text(encoding="utf-8")
    if ext == ".pdf":
        with pymupdf.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    if ext == ".csv":
        return pd.read_csv(path)["text"].astype(str).tolist()
    raise ValueError(f"Unsupported format: {ext}")

def clean_indic(text):
    text = unicodedata.normalize("NFC", text)
    # collapse a combining mark (vowel sign, anusvara) repeated twice in a row
    text = re.sub(r"([\u0900-\u097F\u0C00-\u0C7F])\1+",
                  lambda m: m.group(1) if unicodedata.category(m.group(1)).startswith("M") else m.group(0),
                  text)
    return re.sub(r"[ \t]+", " ", text).strip()

def load_document(path):
    """Extract text from txt/pdf, clean it, and detect its language."""
    raw = extract_text(path)
    if isinstance(raw, list):          # csv returns a list of texts
        return [{"text": clean_text(t), "language": detect_language(t)} for t in raw]
    cleaned = clean_text(raw, from_pdf=str(path).lower().endswith(".pdf"))
    return {"text": cleaned, "language": detect_language(cleaned)}