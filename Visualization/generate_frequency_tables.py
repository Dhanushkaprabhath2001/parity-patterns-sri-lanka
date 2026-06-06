import pandas as pd

# 1. Load Data
df = pd.read_csv('merged_births_2000_2020.csv')

# 2. Standardize Parity for the table
parity_map = {
    'First': '1st', 'Second': '2nd', 'Third': '3rd', 
    'Fourth': '4th', 'Fifth': '5th', 'Sixth': '6th+', 
    'Seventh': '6th+', 'Eighth': '6th+', 'Nineth': '6th+'
}
df['Parity'] = df['Birth_Order'].map(parity_map)

# 3. List of variables to generate frequency tables for
variables = ['Gender', 'Hospital or Not', 'Marital_Status', 'Race_of_Mother']

# Create a writer to save multiple tables to one CSV/Text file or individual ones
# We'll save a combined summary report
with open('frequency_tables_summary.txt', 'w') as f:
    f.write("=== FREQUENCY TABLES ACROSS ALL YEARS (2000-2020) ===\n\n")
    
    for var in variables:
        f.write(f"--- Frequency Table: {var} ---\n")
        # Handle trailing spaces in 'Gender ' if any (though already cleaned in merge)
        counts = df[var].value_counts(dropna=False)
        pcts = df[var].value_counts(normalize=True, dropna=False) * 100
        
        table = pd.DataFrame({'Count': counts, 'Percentage (%)': pcts})
        f.write(table.to_string())
        f.write("\n\n")
        
        # Also save as individual CSV for spreadsheet use
        table.to_csv(f'freq_{var.replace(" ", "_").lower()}.csv')

    # Special Table for Parity across years (Pivot Table style)
    f.write("--- Frequency Table: Parity by Year (Counts) ---\n")
    parity_counts = df.groupby(['Registered_Year', 'Parity']).size().unstack(fill_value=0)
    # Reorder columns
    cols = ['1st', '2nd', '3rd', '4th', '5th', '6th+']
    parity_counts = parity_counts[cols]
    f.write(parity_counts.to_string())
    f.write("\n\n")
    
    f.write("--- Frequency Table: Parity by Year (Percentages %) ---\n")
    parity_pcts = parity_counts.div(parity_counts.sum(axis=1), axis=0) * 100
    f.write(parity_pcts.to_string())
    f.write("\n")

# Save the detailed Parity Pivot as a CSV
parity_counts.to_csv('parity_frequency_counts_by_year.csv')

print("Frequency tables generated:")
print("- frequency_tables_summary.txt (Text report)")
print("- parity_frequency_counts_by_year.csv")
for var in variables:
    print(f"- freq_{var.replace(' ', '_').lower()}.csv")
