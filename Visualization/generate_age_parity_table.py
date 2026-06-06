import pandas as pd
import os
import shutil

# 1. Load Data
df = pd.read_csv('merged_births_2000_2020.csv')

# 2. Data Cleaning & Categorization
# Filter maternal age to realistic range
df = df[(df['Age of Mother'] >= 10) & (df['Age of Mother'] <= 60)]

# Categorize Age
bins = [0, 20, 25, 30, 35, 100]
age_labels = ['<20', '20-24', '25-29', '30-34', '35+']
df['Age_Category'] = pd.cut(df['Age of Mother'], bins=bins, labels=age_labels, right=False)

# Standardize Parity
parity_map = {
    'First': '1st', 'Second': '2nd', 'Third': '3rd', 
    'Fourth': '4th', 'Fifth': '5th', 'Sixth': '6th+', 
    'Seventh': '6th+', 'Eighth': '6th+', 'Nineth': '6th+'
}
df['Parity'] = df['Birth_Order'].map(parity_map)
parity_order = ['1st', '2nd', '3rd', '4th', '5th', '6th+']

# 3. Create Unified Cross-tabulation (Multi-indexed by Year)
# We want Year and Age_Category as index/columns
ct_list = []
years = sorted(df['Registered_Year'].unique())

for year in years:
    year_data = df[df['Registered_Year'] == year]
    # Create cross-tab for this year
    ct = pd.crosstab(year_data['Age_Category'], year_data['Parity'], margins=True, margins_name='Total')
    # Reorder columns
    cols = [p for p in parity_order if p in ct.columns] + ['Total']
    ct = ct.reindex(index=age_labels + ['Total'], columns=cols)
    
    # Calculate row percentages (Parity distribution within each Age Group)
    ct_pct = ct.div(ct['Total'], axis=0) * 100
    
    # Prefix index with year for easy identification in unified table
    ct.index = pd.MultiIndex.from_product([[year], ct.index], names=['Year', 'Age_Group'])
    ct_pct.index = pd.MultiIndex.from_product([[year], ct_pct.index], names=['Year', 'Age_Group'])
    
    ct_list.append(ct)

# Combine all years into one master table
unified_age_parity_ct = pd.concat(ct_list)

# 4. Save Results
output_csv = 'unified_age_parity_crosstab.csv'
unified_age_parity_ct.to_csv(output_csv)

# Generate a clean text report for the user
report_path = 'age_parity_cross_tabulation_report.txt'
with open(report_path, 'w') as f:
    f.write("=== CROSS-TABULATION: MATERNAL AGE GROUP BY PARITY (2000-2020) ===\n\n")
    f.write("This table shows absolute counts for each year.\n\n")
    f.write(unified_age_parity_ct.to_string())
    f.write("\n\nNote: Row totals represent the total births within that age group for that year.\n")

# Copy to results folder
if os.path.exists('Chapter_4_Results'):
    shutil.copy(output_csv, 'Chapter_4_Results/' + output_csv)
    shutil.copy(report_path, 'Chapter_4_Results/' + report_path)

print(f"Cross-tabulation tables generated and saved to {output_csv}")
