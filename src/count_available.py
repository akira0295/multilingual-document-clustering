import pandas as pd

MAPS = {
    "English": ("data/raw/english_lpc_test.csv",
                {"Business": "business", "Education": "education", "Entertainment": "entertainment",
                 "Sports": "sports", "Technology": "technology"}),
    "Hindi":   ("data/raw/hindi_lpc_test.csv",
                {"business": "business", "education": "education", "entertainment": "entertainment",
                 "khel": "sports", "technology-news": "technology"}),
    "Telugu":  ("data/raw/telugu_lpc_test.csv",
                {"business": "business", "education": "education", "entertainment": "entertainment",
                 "sports": "sports", "tech": "technology"}),
}

rows = []
for lang, (path, label_map) in MAPS.items():
    df = pd.read_csv(path).dropna(subset=["text", "labels"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 200].drop_duplicates("text")
    df = df[df["labels"].isin(label_map)]
    counts = df["labels"].map(label_map).value_counts()
    for topic, n in counts.items():
        rows.append({"language": lang, "topic": topic, "available": n})

print(pd.DataFrame(rows).pivot(index="topic", columns="language", values="available"))