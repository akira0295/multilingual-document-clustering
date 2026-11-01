import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/processed/final_dataset_clean.csv")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print("Model max_seq_length:", model.max_seq_length)

df["n_tokens"] = df["clean_text"].apply(lambda t: len(model.tokenizer.tokenize(t)))
limit = model.max_seq_length
summary = df.groupby("language")["n_tokens"].agg(["median", "max"])
summary["pct_over_limit"] = df.groupby("language")["n_tokens"].apply(lambda s: round((s > limit).mean() * 100, 1))
print(summary)