import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("results", exist_ok=True)
df = pd.read_csv("data/processed/final_dataset_clean.csv")
emb = np.load("data/processed/final_embeddings_LaBSE.npy")

xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42).fit_transform(emb)
df["x"], df["y"] = xy[:, 0], xy[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
sns.scatterplot(data=df, x="x", y="y", hue="topic", s=12, ax=axes[0])
axes[0].set_title("LaBSE embeddings by topic")
sns.scatterplot(data=df, x="x", y="y", hue="language", s=12, ax=axes[1])
axes[1].set_title("LaBSE embeddings by language")
for ax in axes:
    ax.set_xlabel(""); ax.set_ylabel("")
plt.tight_layout()
plt.savefig("results/umap_labse.png", dpi=200)
print("Saved results/umap_labse.png")