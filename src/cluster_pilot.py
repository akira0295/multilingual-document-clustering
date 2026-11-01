import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

df = pd.read_csv("data/processed/pilot_dataset.csv")
emb = np.load("data/processed/pilot_embeddings.npy")
assert len(df) == len(emb) == 750

# Joint clustering: all three languages together, no splitting by language
# Ward linkage can't use cosine, so use average linkage
model = AgglomerativeClustering(n_clusters=5, metric="cosine", linkage="average")
df["cluster"] = model.fit_predict(emb)

# Evaluation against topic labels (labels were NOT used to form clusters)
print("Silhouette (cosine):", round(silhouette_score(emb, df["cluster"], metric="cosine"), 4))
print("ARI:", round(adjusted_rand_score(df["topic"], df["cluster"]), 4))
print("NMI:", round(normalized_mutual_info_score(df["topic"], df["cluster"]), 4))

# Do clusters follow topic, or language?
print("\nCluster vs topic:")
print(pd.crosstab(df["cluster"], df["topic"]))
print("\nCluster vs language:")
print(pd.crosstab(df["cluster"], df["language"]))

# Per-language ARI shows where Telugu falls behind
for lang, g in df.groupby("language"):
    print(lang, "ARI:", round(adjusted_rand_score(g["topic"], g["cluster"]), 4))

df.to_csv("data/processed/pilot_clustered.csv", index=False)