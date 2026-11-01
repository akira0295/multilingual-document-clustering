import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

df = pd.read_csv("data/processed/final_dataset_clean.csv")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
tok, LIMIT = model.tokenizer, 120   # leave room for special tokens

def chunk(text):
    ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
    pieces = [ids[i:i + LIMIT] for i in range(0, len(ids), LIMIT)]
    return [tok.decode(p) for p in pieces] or [text]


# 1) Truncated baseline (first 128 tokens only)
emb_trunc = model.encode(df["clean_text"].tolist(), batch_size=32,
                         show_progress_bar=True, normalize_embeddings=True)

# 2) Chunked: embed every chunk, average per document
all_chunks, owner = [], []
for i, t in enumerate(df["clean_text"]):
    for c in chunk(t):
        all_chunks.append(c); owner.append(i)
owner = np.array(owner)
print("Total chunks:", len(all_chunks), "for", len(df), "documents")
print("Chunks per document:", round(len(all_chunks) / len(df), 2))
ce = model.encode(all_chunks, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
emb_chunk = np.vstack([ce[owner == i].mean(axis=0) for i in range(len(df))])
emb_chunk /= np.linalg.norm(emb_chunk, axis=1, keepdims=True)

np.save("data/processed/final_embeddings_trunc.npy", emb_trunc)
np.save("data/processed/final_embeddings_chunk.npy", emb_chunk)

def evaluate(name, emb, labels):
    topic = df["topic"]
    row = {"setup": name,
           "silhouette": round(silhouette_score(emb, labels, metric="cosine"), 3),
           "ARI": round(adjusted_rand_score(topic, labels), 3),
           "NMI": round(normalized_mutual_info_score(topic, labels), 3)}
    for lang in ["English", "Hindi", "Telugu"]:
        m = (df["language"] == lang).values
        row[f"ARI_{lang[:2]}"] = round(adjusted_rand_score(topic[m], labels[m]), 3)
    return row

rows = []
for ename, emb in [("trunc", emb_trunc), ("chunk", emb_chunk)]:
    agg = AgglomerativeClustering(n_clusters=5, metric="cosine", linkage="average").fit_predict(emb)
    km = KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(emb)
    rows.append(evaluate(f"{ename}+agglomerative", emb, agg))
    rows.append(evaluate(f"{ename}+kmeans", emb, km))

print(pd.DataFrame(rows).to_string(index=False))