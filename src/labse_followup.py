import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

df = pd.read_csv("data/processed/final_dataset_clean.csv")
emb = np.load("data/processed/final_embeddings_LaBSE.npy")
topic = df["topic"]

def report(name, labels):
    row = {"setup": name,
           "silhouette": round(silhouette_score(emb, labels, metric="cosine"), 3),
           "ARI": round(adjusted_rand_score(topic, labels), 3),
           "NMI": round(normalized_mutual_info_score(topic, labels), 3)}
    for lang in ["English", "Hindi", "Telugu"]:
        m = (df["language"] == lang).values
        row[f"ARI_{lang[:2]}"] = round(adjusted_rand_score(topic[m], labels[m]), 3)
    return row

rows = []
for link in ["average", "complete"]:
    lab = AgglomerativeClustering(n_clusters=5, metric="cosine", linkage=link).fit_predict(emb)
    rows.append(report(f"LaBSE+agglomerative-{link}", lab))
km = KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(emb)
rows.append(report("LaBSE+kmeans", km))
print(pd.DataFrame(rows).to_string(index=False))

# English diagnostic: which topics get mixed?
en = (df["language"] == "English").values
print("\nEnglish only, cluster vs topic (KMeans):")
print(pd.crosstab(km[en], topic[en]))
