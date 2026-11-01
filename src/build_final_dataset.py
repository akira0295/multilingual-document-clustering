import urllib.request
import pandas as pd
import os
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

BASE = "https://raw.githubusercontent.com/l3cube-pune/indic-nlp/main/L3Cube-IndicNews"
SOURCES = {
    "English": ("English/LPC/English_test_lpc.csv", "EN",
                {"Business": "business", "Education": "education",
                 "Entertainment": "entertainment", "Sports": "sports",
                 "Technology": "technology"}),
    "Hindi":   ("Hindi/LPC/Hindi_LPC_Test.csv", "HI",
                {"business": "business", "education": "education",
                 "entertainment": "entertainment", "khel": "sports",
                 "technology-news": "technology"}),
    "Telugu":  ("Telugu/LPC/Telugu_LPC_Test.csv", "TE",
                {"business": "business", "education": "education",
                 "entertainment": "entertainment", "sports": "sports",
                 "tech": "technology"}),
}
N = 150          # documents per topic per language
MIN_CHARS = 200

frames = []
for lang, (path, code, label_map) in SOURCES.items():
    local = f"data/raw/{lang.lower()}_lpc_test.csv"
    urllib.request.urlretrieve(f"{BASE}/{path}", local)
    df = pd.read_csv(local)
    df = df.dropna(subset=["text", "labels"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= MIN_CHARS].drop_duplicates("text")
    df = df[df["labels"].isin(label_map)].copy()
    df["original_label"] = df["labels"]
    df["topic"] = df["labels"].map(label_map)
    df = (pd.concat([df[df.topic == t].sample(N, random_state=42)
                     for t in sorted(set(label_map.values()))])
            .reset_index(drop=True))
    df["language"] = lang
    df["source"] = "L3Cube-IndicNews-LPC"
    df["doc_id"] = (code + "_" + df["topic"].str.upper() + "_"
                    + (df.groupby("topic").cumcount() + 1).astype(str).str.zfill(3))
    frames.append(df[["doc_id", "language", "topic", "text", "source",
                      "original_label"]])

out = pd.concat(frames, ignore_index=True)
out.to_csv("data/processed/final_dataset.csv", index=False)
print(pd.crosstab(out["topic"], out["language"]))
print(out.groupby("language")["text"].apply(lambda s: s.str.len().median()))
