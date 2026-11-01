from datasets import load_dataset
import pandas as pd

def sample(split, n=200, min_chars=200):
    ds = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2",
                      split=split, streaming=True)
    rows = []
    for row in ds:
        t = row["text"].strip()
        if len(t) >= min_chars:
            rows.append({"language": split, "text": t})
        if len(rows) == n:
            break
    return pd.DataFrame(rows)

df = pd.concat([sample("tel_Telu"), sample("hin_Deva")], ignore_index=True)
df.to_csv("data/raw/indiccorp_sample.csv", index=False)
print(df.groupby("language").size())
print(df["text"].str.len().describe())
