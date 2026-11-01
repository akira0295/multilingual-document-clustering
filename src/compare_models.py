import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

name = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/LaBSE"
df = pd.read_csv("data/processed/final_dataset_clean.csv")
model = SentenceTransformer(name)
tok, LIMIT = model.tokenizer, model.max_seq_length - 8
print("Model:", name, "| max_seq_length:", model.max_seq_length)

def chunk(text):
    ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
    pieces = [ids[i:i + LIMIT] for i in range(0, len(ids), LIMIT)]
    return [tok.decode(p) for p in pieces] or [text]

chunks, owner = [], []
for i, t in enumerate(df["clean_text"]):
    for c in chunk(t):
        chunks.append(c); owner.append(i)
owner = np.array(owner)
print("Chunks per document:", round(len(chunks) / len(df), 2))

ce = model.encode(chunks, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
emb = np.vstack([ce[owner == i].mean(axis=0) for i in range(len(df))])
emb /= np.linalg.norm(emb, axis=1, keepdims=True)
tag = name.split("/")[-1]
np.save(f"data/processed/final_embeddings_{tag}.npy", emb)

topic = df["topic"]
for seed in [0, 1, 2]:
    km = KMeans(n_clusters=5, n_init=10, random_state=seed).fit_predict(emb)
    row = {"seed": seed,
           "silhouette": round(silhouette_score(emb, km, metric="cosine"), 3),
           "ARI": round(adjusted_rand_score(topic, km), 3),
           "NMI": round(normalized_mutual_info_score(topic, km), 3)}
    for lang in ["English", "Hindi", "Telugu"]:
        m = (df["language"] == lang).values
        row[f"ARI_{lang[:2]}"] = round(adjusted_rand_score(topic[m], km[m]), 3)
    print(row)

print("\nCluster vs language (last seed):")
print(pd.crosstab(km, df["language"]))
print("\nTelugu only, cluster vs topic:")
print(pd.crosstab(km[(df["language"] == "Telugu").values], topic[df["language"] == "Telugu"]))