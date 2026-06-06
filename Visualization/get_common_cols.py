import pandas as pd

files = [
    'complete_birth_2010.xlsx',
    'complete_birth_2015.xlsx',
    'complete_birth_2020.xlsx'
]

columns_sets = []

for file in files:
    try:
        # Reading only the first row to get column names efficiently
        df = pd.read_excel(file, nrows=0)
        columns_sets.append(set(df.columns))
        print(f"Columns in {file}: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading {file}: {e}")

if columns_sets:
    common_columns = set.intersection(*columns_sets)
    print("\nCommon Variables:")
    print(sorted(list(common_columns)))
