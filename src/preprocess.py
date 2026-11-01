import re
import unicodedata
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # makes langdetect deterministic

def clean_text(text, from_pdf=False):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    if from_pdf:
        text = re.sub(
            r"([\u0900-\u097F\u0C00-\u0C7F])\1+",
            lambda m: m.group(1) if unicodedata.category(m.group(1)).startswith("M") else m.group(0),
            text,
        )
    return re.sub(r"\s+", " ", text).strip()

def detect_language(text):
    try:
        return {"en": "English", "hi": "Hindi", "te": "Telugu"}.get(detect(text), "Other")
    except Exception:
        return "Other"