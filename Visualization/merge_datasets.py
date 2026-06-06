import pandas as pd
import os

files = {
    2000: 'complete_birth_2000.xlsx',
    2005: 'complete_birth_2005.xlsx',
    2010: 'complete_birth_2010.xlsx',
    2015: 'complete_birth_2015.xlsx',
    2020: 'complete_birth_2020.xlsx'
}

# Expanded common columns
common_cols = [
    'Registered_Year', 'Registered_Month', 'Registered_District',
    'Birh_Year', 'Birth_Month', 'Gender', 'Hospital or Not',
    'Birth_Order', 'Age of Mother', 'Marital_Status',
    'District_of_Mother', 'Race_of_Mother', 'Multiple_Birth_Status'
]

all_dfs = []

for year, file_path in files.items():
    print(f"Processing {file_path}...")
    df = pd.read_excel(file_path)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    
    if 'Gender ' in df.columns and 'Gender' not in df.columns:
        df.rename(columns={'Gender ': 'Gender'}, inplace=True)
    
    # Check for Multiple_Birth_Status
    if 'Multiple_Birth_Status' not in df.columns:
        df['Multiple_Birth_Status'] = 'Unknown/Not Recorded'

    available_cols = [c for c in common_cols if c in df.columns]
    all_dfs.append(df[available_cols])

merged_df = pd.concat(all_dfs, ignore_index=True)
merged_df.to_csv('merged_births_2000_2020.csv', index=False)
print(f"Successfully re-merged with Multiple_Birth_Status. Total records: {len(merged_df)}")
