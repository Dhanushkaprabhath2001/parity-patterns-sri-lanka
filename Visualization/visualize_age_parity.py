import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv('merged_births_2000_2020.csv')

# 2. Data Cleaning
# Filter maternal age to a realistic range (12 to 55) to remove outliers like 999
df = df[(df['Age of Mother'] >= 12) & (df['Age of Mother'] <= 55)]

# Standardize Parity
parity_map = {
    'First': '1st', 'Second': '2nd', 'Third': '3rd', 
    'Fourth': '4th', 'Fifth': '5th', 'Sixth': '6th+', 
    'Seventh': '6th+', 'Eighth': '6th+', 'Nineth': '6th+'
}
df['Parity'] = df['Birth_Order'].map(parity_map)

# Define logical order for plotting
parity_order = ['1st', '2nd', '3rd', '4th', '5th', '6th+']
years = sorted(df['Registered_Year'].unique())

# 3. Visualization
plt.figure(figsize=(15, 10))
sns.set_style("whitegrid")

# Create a combined boxplot showing how the age distribution for each parity shifts over years
# We'll use Year as the hue to see the "Shift" directly in one plot
ax = sns.boxplot(x='Parity', y='Age of Mother', hue='Registered_Year', 
                 data=df, palette='viridis', order=parity_order)

plt.title('Maternal Age Distribution by Birth Order and Year (2000-2020)', fontsize=18, fontweight='bold')
plt.xlabel('Birth Order (Parity)', fontsize=14, fontweight='bold')
plt.ylabel('Maternal Age (Years)', fontsize=14, fontweight='bold')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add horizontal lines for median age of 1st parity in 2000 vs 2020 to highlight the shift
median_2000_1st = df[(df['Registered_Year'] == 2000) & (df['Parity'] == '1st')]['Age of Mother'].median()
median_2020_1st = df[(df['Registered_Year'] == 2020) & (df['Parity'] == '1st')]['Age of Mother'].median()

print(f"Median age for 1st birth in 2000: {median_2000_1st}")
print(f"Median age for 1st birth in 2020: {median_2020_1st}")

plt.tight_layout()
output_img = 'maternal_age_by_parity_boxplot.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Visualization saved to {output_img}")

# Also calculate a summary table for the thesis
summary_stats = df.groupby(['Registered_Year', 'Parity'])['Age of Mother'].agg(['median', 'mean', 'std']).reset_index()
summary_stats.to_csv('maternal_age_parity_summary.csv', index=False)
print("Summary statistics saved to maternal_age_parity_summary.csv")
