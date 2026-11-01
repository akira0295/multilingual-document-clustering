import numpy as np
import pandas as pd

df = pd.read_csv("data/processed/final_dataset_clean.csv")
emb = np.load("data/processed/final_embeddings_LaBSE.npy")
topic = df["topic"].to_numpy(dtype=object)
lang = df["language"].to_numpy(dtype=object)
langs, K = ["English", "Hindi", "Telugu"], 10
sims = emb @ emb.T

# 1) Precision@10 by query language and target language
prec = pd.DataFrame(index=langs, columns=langs, dtype=float)
for ql in langs:
    qi = np.where(lang == ql)[0]
    for tl in langs:
        ti = np.where(lang == tl)[0]
        S = sims[np.ix_(qi, ti)].copy()
        if ql == tl:
            np.fill_diagonal(S, -1)          # exclude the query itself
        top = np.argsort(-S, axis=1)[:, :K]
        prec.loc[ql, tl] = (topic[ti][top] == topic[qi][:, None]).mean()
print("Precision@10 (rows = query language, columns = searched language):")
print(prec.round(3))

# 2) Language share of the top 10 when searching all documents
share = pd.DataFrame(index=langs, columns=langs, dtype=float)
for ql in langs:
    qi = np.where(lang == ql)[0]
    S = sims[qi].copy()
    S[np.arange(len(qi)), qi] = -1
    top = np.argsort(-S, axis=1)[:, :K]
    for tl in langs:
        share.loc[ql, tl] = (lang[top] == tl).mean()
print("\nShare of top-10 results by language (chance is about 0.33 each):")
print(share.round(3))