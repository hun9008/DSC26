import pandas as pd
import glob

for path in glob.glob("./*_submission*.csv"):
    df = pd.read_csv(path)
    count = df["decision"].sum()
    print(path, "→ True 개수 =", count)