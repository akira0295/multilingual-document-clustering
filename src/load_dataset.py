from datasets import load_dataset

tel = load_dataset(
    "ai4bharat/IndicCorpV2",
    "indiccorp_v2",
    split="tel_Telu",
    streaming=True,
)

for i, row in enumerate(tel):
    print(row)
    if i == 4:
        break
