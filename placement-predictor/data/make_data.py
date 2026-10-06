import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000

df = pd.DataFrame({
    "cgpa": rng.normal(7.2, 1.0, n).clip(4, 10).round(2),
    "projects": rng.integers(0, 6, n),
    "internships": rng.integers(0, 4, n),
    "coding_score": rng.integers(20, 100, n),
    "communication": rng.integers(1, 11, n),
})

score = (
    0.9 * (df.cgpa - 7)
    + 0.4 * df.projects
    + 0.6 * df.internships
    + 0.03 * (df.coding_score - 60)
    + 0.15 * (df.communication - 5)
    + rng.normal(0, 0.8, n)
)
df["placed"] = (score > 0.8).astype(int)

df.to_csv("data/placement.csv", index=False)
print(df.shape)
print(df["placed"].value_counts())