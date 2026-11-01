import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/raw/indiccorp_sample.csv")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

emb = model.encode(df["text"].tolist(), batch_size=32,
                   show_progress_bar=True, normalize_embeddings=True)
np.save("data/processed/embeddings.npy", emb)
print("Embeddings shape:", emb.shape)

# Sanity check: for 3 Hindi docs, show the closest Telugu doc
hi = df.index[df.language == "hin_Deva"][:3]
te = df.index[df.language == "tel_Telu"]
for i in hi:
    sims = emb[te] @ emb[i]
    j = te[sims.argmax()]
    print("\nHINDI :", df.text[i][:150])
    print("TELUGU:", df.text[j][:150])
    print("cosine:", round(float(sims.max()), 3))
