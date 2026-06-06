import pandas as pd

files = {
    2010: 'complete_birth_2010.xlsx',
    2015: 'complete_birth_2015.xlsx',
    2020: 'complete_birth_2020.xlsx'
}

dfs = []

# Define the target common columns (including Gender after cleanup)
common_cols = [
    'Age of Mother', 'Birh_Year', 'Birth_Month', 'Birth_Order', 
    'Birth_Weight(grams)', 'District_of_Mother', 'Gender', 
    'Hospital or Not', 'Marital_Status', 'Multiple_Birth_Status', 
    'Race_of_Father', 'Race_of_Mother', 'Registered_District', 
    'Registered_Month', 'Registered_Year'
]

for year, file in files.items():
    print(f"Processing {file}...")
    try:
        # Read the file
        df = pd.read_excel(file)
        
        # Strip whitespace from column names to handle 'Gender ' vs 'Gender'
        df.columns = df.columns.str.strip()
        
        # Select only the common columns that exist in the dataframe
        # (This handles the case if any were slightly different but now stripped)
        df_filtered = df[common_cols]
        
        # Optional: Add a 'Source_Year' column if needed to distinguish records, 
        # though Birh_Year/Registered_Year might already cover this.
        
        dfs.append(df_filtered)
        print(f"Successfully processed {len(df_filtered)} rows from {year}.")
    except Exception as e:
        print(f"Error processing {file}: {e}")

if dfs:
    merged_df = pd.concat(dfs, ignore_index=True)
    output_file = 'merged_2010_2015_2020.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\nMerged data saved to {output_file}")
    print(f"Total rows: {len(merged_df)}")
    print(f"Columns: {list(merged_df.columns)}")
