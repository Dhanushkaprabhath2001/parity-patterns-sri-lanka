import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

# 1. Load Data
df = pd.read_csv('merged_births_2000_2020.csv')

# 2. Filter to years with Multiple Birth data
# We'll use 2010, 2015, and 2020
df_multi = df[df['Multiple_Birth_Status'] != 'Unknown/Not Recorded'].copy()

# 3. Standardize Parity
parity_map = {
    'First': '1st', 'Second': '2nd', 'Third': '3rd', 
    'Fourth': '4th', 'Fifth': '5th', 'Sixth': '6th+', 
    'Seventh': '6th+', 'Eighth': '6th+', 'Nineth': '6th+'
}
df_multi['Parity'] = df_multi['Birth_Order'].map(parity_map)
parity_order = ['1st', '2nd', '3rd', '4th', '5th', '6th+']

# 4. Calculate Multiple Birth Rate (MBR) per 100 births
# MBR = (Twins + Triplets + etc) / Total Births * 100
def is_multiple(val):
    return 1 if val in ['Twin', 'Triplets', 'Quadruplets', 'Quintuplets'] else 0

df_multi['Is_Multiple'] = df_multi['Multiple_Birth_Status'].apply(is_multiple)

# Group by Year and Parity
mbr_stats = df_multi.groupby(['Registered_Year', 'Parity'])['Is_Multiple'].agg(['count', 'sum']).reset_index()
mbr_stats['MBR_Percentage'] = (mbr_stats['sum'] / mbr_stats['count'] * 100).round(3)

# Ensure parity is ordered for plotting
mbr_stats['Parity'] = pd.Categorical(mbr_stats['Parity'], categories=parity_order, ordered=True)
mbr_stats = mbr_stats.sort_values(['Registered_Year', 'Parity'])

# 5. Professional Visualization: Line Plot showing Trend
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--'})

# Plotting each year as a separate line to show the consistency of the parity effect
palette = sns.color_palette("Set1", n_colors=3)
ax = sns.lineplot(data=mbr_stats, x='Parity', y='MBR_Percentage', hue='Registered_Year', 
                  marker='o', markersize=10, linewidth=2.5, palette=palette)

# Aesthetics
plt.title('Trend in Multiple Birth Rate by Parity (2010-2020)', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('Multiple Births per 100 Total Births (%)', fontsize=14, fontweight='bold')
plt.xlabel('Birth Order (Parity)', fontsize=14, fontweight='bold')
plt.ylim(0, 2.5) # Providing a stable, professional scale
plt.legend(title='Study Year', title_fontsize=12, fontsize=11, loc='upper left')

# Add data labels for the most recent year (2020) to provide immediate context
year_2020 = mbr_stats[mbr_stats['Registered_Year'] == 2020]
for x, y in zip(year_2020['Parity'], year_2020['MBR_Percentage']):
    plt.annotate(f'{y:.2f}%', xy=(x, y), xytext=(0, 12), 
                 textcoords='offset points', ha='center', fontsize=10, fontweight='bold', color=palette[2])

plt.tight_layout()
output_img = 'parity_multiple_birth_trend.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')

# 6. Save a clean data table for the thesis
mbr_pivot = mbr_stats.pivot(index='Parity', columns='Registered_Year', values='MBR_Percentage')
mbr_pivot.to_csv('multiple_birth_rates_by_parity.csv')

# Copy to results
if os.path.exists('Chapter_4_Results'):
    shutil.copy(output_img, 'Chapter_4_Results/' + output_img)
    mbr_pivot.to_csv('Chapter_4_Results/multiple_birth_rates_by_parity.csv')

print(f"Professional trend plot saved to {output_img}")
print("Statistical table saved to multiple_birth_rates_by_parity.csv")
