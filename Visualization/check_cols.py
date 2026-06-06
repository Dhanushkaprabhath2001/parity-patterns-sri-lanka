import pandas as pd
import os

years = [2000, 2005, 2010, 2015, 2020]
for y in years:
    f = f'complete_birth_{y}.xlsx'
    if os.path.exists(f):
        try:
            df = pd.read_excel(f, nrows=1)
            print(f"--- {f} ---")
            print(df.columns.tolist())
        except Exception as e:
            print(f"Error reading {f}: {e}")
    else:
        print(f"{f} not found")
