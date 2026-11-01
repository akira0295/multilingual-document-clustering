from datasets import load_dataset
import pandas as pd

ds = load_dataset("fancyzhx/ag_news", split="train")
df = ds.to_pandas()

# AG labels: 0 World, 1 Sports, 2 Business, 3 Sci/Tech
topic_map = {0: "politics", 1: "sports", 2: "business"}
df = df[df["label"].isin(topic_map)].copy()
df["topic"] = df["label"].map(topic_map)

N = 100  # documents per topic
parts = [df[df["topic"] == t].sample(N, random_state=42)
         for t in topic_map.values()]
df = pd.concat(parts, ignore_index=True)

df["language"] = "English"
df["source"] = "ag_news"
df["doc_id"] = (
    "EN_" + df["topic"].str.upper() + "_"
    + (df.groupby("topic").cumcount() + 1).astype(str).str.zfill(3)
)
df = df[["doc_id", "language", "topic", "text", "source"]]

df.to_csv("data/raw/english_labeled.csv", index=False)
print(df.groupby("topic").size())
print(df["text"].str.len().describe())
print(df.head(3).to_string())