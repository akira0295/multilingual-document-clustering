import pandas as pd

df = pd.read_csv("data/raw/telugu_news_train.csv")
print("Columns:", list(df.columns))
print(df["topic"].value_counts())

topic_map = {"business": "business", "sports": "sports",
             "sport": "sports", "nation": "politics"}
df = df[df["topic"].isin(topic_map)].copy()
df["topic"] = df["topic"].map(topic_map)

df["text"] = (df["heading"].fillna("") + ". " + df["body"].fillna("")).str.strip()
df["text"] = df["text"].str.slice(0, 1500)
df = df[df["text"].str.len() >= 200].drop_duplicates(subset="text")

N = 100
parts = [df[df["topic"] == t].sample(N, random_state=42)
         for t in ["business", "sports", "politics"]]
df = pd.concat(parts, ignore_index=True)

df["language"] = "Telugu"
df["source"] = "telugu_news"
df["doc_id"] = (
    "TE_" + df["topic"].str.upper() + "_"
    + (df.groupby("topic").cumcount() + 1).astype(str).str.zfill(3)
)
df = df[["doc_id", "language", "topic", "text", "source"]]

df.to_csv("data/raw/telugu_labeled.csv", index=False)
print(df.groupby("topic").size())
print(df["text"].str.len().describe())
print(df.head(2).to_string())
