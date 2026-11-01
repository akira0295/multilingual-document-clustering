from extract_text import load_document

for name in ["english", "hindi", "telugu"]:
    doc = load_document(f"data/raw/pdf_test/{name}.pdf")
    print(f"\n=== {name.upper()} ===")
    print("Detected language:", doc["language"])
    print("Characters:", len(doc["text"]))
    print(doc["text"][:200])