import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/processed/pilot_dataset.csv")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
emb = model.encode(df["text"].tolist(), batch_size=32,
                   show_progress_bar=True, normalize_embeddings=True)
np.save("data/processed/pilot_embeddings.npy", emb)

langs = ["English", "Hindi", "Telugu"]
topics = sorted(df["topic"].unique())

def centroid(lang, topic):
    c = emb[((df.language == lang) & (df.topic == topic)).values].mean(axis=0)
    return c / np.linalg.norm(c)

print("\nCosine similarity between language centroids, same topic:")
rows = []
for t in topics:
    row = {"topic": t}
    for a, b in [("English", "Hindi"), ("English", "Telugu"), ("Hindi", "Telugu")]:
        row[f"{a[:2]}-{b[:2]}"] = round(float(centroid(a, t) @ centroid(b, t)), 3)
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

# Same-topic vs different-topic similarity across languages
same, diff = [], []
for a, b in [("English", "Hindi"), ("English", "Telugu"), ("Hindi", "Telugu")]:
    for ta in topics:
        for tb in topics:
            s = float(centroid(a, ta) @ centroid(b, tb))
            (same if ta == tb else diff).append(s)
print("\nAverage centroid similarity, same topic across languages:", round(np.mean(same), 3))
print("Average centroid similarity, different topics across languages:", round(np.mean(diff), 3))
