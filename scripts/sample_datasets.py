import pandas as pd

for name in ["gsm8k", "logiqa", "commonsenseqa"]:
    df = pd.read_csv(f"data/processed/{name}.csv")
    sample = df.sample(n=min(150, len(df)), random_state=42)
    sample.to_csv(f"data/processed/{name}_sample.csv", index=False)
    print(f"{name}: sampled {len(sample)} / {len(df)}")
