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

# 3. Process each year
years = sorted(df['Registered_Year'].unique())
parity_order = ['1st', '2nd', '3rd', '4th', '5th', '6th+']

with open('parity_gender_frequency_report.txt', 'w') as f:
    f.write("=== FREQUENCY TABLE: PARITY BY GENDER FOR EACH YEAR ===\n\n")

    for year in years:
        year_data = df[df['Registered_Year'] == year]
        
        # Create Cross-tabulation (Pivot Table)
        # We use dropna=False to see if there are any missing values
        ct = pd.crosstab(year_data['Parity'], year_data['Gender'], margins=True, margins_name='Total')
        
        # Reorder rows to match logical parity sequence
        existing_rows = [p for p in parity_order if p in ct.index] + ['Total']
        ct = ct.reindex(existing_rows)
        
        # Calculate Row Percentages (Gender distribution within each parity)
        ct_pct = ct.div(ct['Total'], axis=0) * 100
        
        f.write(f"--- YEAR: {year} ---\n")
        f.write("Counts:\n")
        f.write(ct.to_string())
        f.write("\n\nGender Distribution Within Parity (%):\n")
        f.write(ct_pct.round(2).to_string())
        f.write("\n" + "="*40 + "\n\n")
        
        # Save each year to a separate CSV for easy access
        ct.to_csv(f'parity_gender_freq_{year}.csv')

print("Frequency tables generated:")
print("- parity_gender_frequency_report.txt (Combined Text Report)")
for year in years:
    print(f"- parity_gender_freq_{year}.csv")
