import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/processed/final_dataset_clean.csv")
emb = np.load("data/processed/final_embeddings_LaBSE.npy")
model = SentenceTransformer("sentence-transformers/LaBSE")

def search(query, k=10):
    q = model.encode([query], normalize_embeddings=True)[0]
    sims = emb @ q                      # cosine similarity (vectors are normalized)
    top = np.argsort(-sims)[:k]
    out = df.iloc[top][["doc_id", "language", "topic", "clean_text"]].copy()
    out["score"] = sims[top].round(3)
    return out

queries = sys.argv[1:] or [
    "cricket match result",
    "क्रिकेट मैच का परिणाम",
    "క్రికెట్ మ్యాచ్ ఫలితం",
]
for q in queries:
    res = search(q)
    print(f"\n=== Query: {q} ===")
    print("Languages in top 10:", res["language"].value_counts().to_dict())
    print("Topics in top 10:", res["topic"].value_counts().to_dict())
    for _, r in res.head(5).iterrows():
        print(f"{r.score} | {r.language} | {r.topic} | {r.doc_id} | {r.clean_text[:80]}")