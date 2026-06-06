import pandas as pd

# 1. Load Data
df = pd.read_csv('merged_births_2000_2020.csv')

# 2. Standardize Parity
parity_map = {
    'First': '1st', 'Second': '2nd', 'Third': '3rd', 
    'Fourth': '4th', 'Fifth': '5th', 'Sixth': '6th+', 
    'Seventh': '6th+', 'Eighth': '6th+', 'Nineth': '6th+'
}
df['Parity'] = df['Birth_Order'].map(parity_map)

# 3. Create Unified Cross-tabulation
# We group by Year, Parity, and Gender to get counts
unified_counts = df.groupby(['Parity', 'Registered_Year', 'Gender']).size().unstack(level=[1, 2], fill_value=0)

# Reorder Parity Index logically
parity_order = ['1st', '2nd', '3rd', '4th', '5th', '6th+']
existing_rows = [p for p in parity_order if p in unified_counts.index]
unified_counts = unified_counts.reindex(existing_rows)

# Add Totals for each Year
years = sorted(df['Registered_Year'].unique())
for year in years:
    unified_counts[(year, 'Total')] = unified_counts[year].sum(axis=1)

# Sort columns by Year and then Gender (Male, Female, Total)
unified_counts = unified_counts.sort_index(axis=1)

# 4. Save to CSV
output_csv = 'unified_parity_gender_crosstab.csv'
unified_counts.to_csv(output_csv)

# 5. Generate a Text Summary version
with open('unified_parity_gender_report.txt', 'w') as f:
    f.write("=== UNIFIED CROSS-TABULATION: PARITY BY GENDER AND YEAR ===\n\n")
    f.write(unified_counts.to_string())
    f.write("\n\n(Columns are Multi-indexed by [Year, Gender])\n")

print(f"Unified table generated: {output_csv}")
print("Text report generated: unified_parity_gender_report.txt")
