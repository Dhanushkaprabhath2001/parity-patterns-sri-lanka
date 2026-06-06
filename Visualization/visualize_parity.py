import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv('merged_births_2000_2020.csv')

# 2. Map Categorical Birth Order to Numeric/Sorted Labels
parity_map = {
    'First': '1st',
    'Second': '2nd',
    'Third': '3rd',
    'Fourth': '4th',
    'Fifth': '5th',
    'Sixth': '6th+',
    'Seventh': '6th+',
    'Eighth': '6th+',
    'Nineth': '6th+'
}

df['Parity'] = df['Birth_Order'].map(parity_map)

# Remove any rows where Parity couldn't be mapped (NaNs)
df = df.dropna(subset=['Parity'])

# 3. Calculate Distribution (Percentages) per Year
# We want to see how the proportion of 1st, 2nd, etc. births changes each year
dist_df = df.groupby(['Registered_Year', 'Parity']).size().unstack(fill_value=0)
# Normalize to percentages
dist_pct = dist_df.div(dist_df.sum(axis=1), axis=0) * 100

# Reorder columns for logical progression
cols = ['1st', '2nd', '3rd', '4th', '5th', '6th+']
dist_pct = dist_pct[cols]

# 4. Visualization
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

# Plotting as a stacked bar chart to show the "Composition" of births each year
ax = dist_pct.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='viridis')

plt.title('Distribution of Birth Parity (Birth Order) by Year', fontsize=16, fontweight='bold')
plt.ylabel('Percentage of Total Births (%)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Parity', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add percentage labels on the bars for clarity
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    if height > 3: # Only label if the slice is large enough to see
        ax.text(x + width/2, 
                y + height/2, 
                f'{height:.1f}%', 
                horizontalalignment='center', 
                verticalalignment='center',
                fontsize=9,
                color='white',
                fontweight='bold')

plt.tight_layout()
output_img = 'parity_distribution_by_year.png'
plt.savefig(output_img, dpi=300)
print(f"Parity distribution chart saved to {output_img}")

# Also save the percentage table for reference
dist_pct.to_csv('parity_distribution_percentages.csv')
print("Percentage data saved to parity_distribution_percentages.csv")
