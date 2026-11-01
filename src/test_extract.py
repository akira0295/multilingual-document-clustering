from extract_text import extract_text

for name in ["english", "hindi", "telugu"]:
    text = extract_text(f"data/raw/pdf_test/{name}.pdf")
    print(f"\n=== {name.upper()} ===")
    print("Characters extracted:", len(text))
    print(text[:300])