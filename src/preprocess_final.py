import pandas as pd
from preprocess import clean_text, detect_language

df = pd.read_csv("data/processed/final_dataset.csv")
df["clean_text"] = df["text"].apply(clean_text)
df["detected_language"] = df["clean_text"].apply(detect_language)

print("Overall agreement:", round((df["detected_language"] == df["language"]).mean() * 100, 2), "%")
print(pd.crosstab(df["language"], df["detected_language"]))
print("Empty after cleaning:", (df["clean_text"].str.len() == 0).sum())

df.to_csv("data/processed/final_dataset_clean.csv", index=False)