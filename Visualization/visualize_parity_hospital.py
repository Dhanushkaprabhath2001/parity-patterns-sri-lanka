import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

# 1. Load Data
df = pd.read_csv('merged_births_2000_2020.csv')

# 2. Standardize Parity
parity_map = {
    'First': '1st', 'Second': '2nd', 'Third': '3rd', 
    'Fourth': '4th', 'Fifth': '5th', 'Sixth': '6th+', 
    'Seventh': '6th+', 'Eighth': '6th+', 'Nineth': '6th+'
}
df['Parity'] = df['Birth_Order'].map(parity_map)

# 3. Clean 'Hospital or Not' labels
df['Place_of_Birth'] = df['Hospital or Not'].replace({'Hospital': 'Hospital', 'Not in Hospital': 'Non-Hospital'})

# Define order for plotting
parity_order = ['1st', '2nd', '3rd', '4th', '5th', '6th+']
years = sorted(df['Registered_Year'].unique())

# 4. Visualization
fig, axes = plt.subplots(len(years), 1, figsize=(14, 25))
sns.set_style("whitegrid")

for i, year in enumerate(years):
    ax = axes[i]
    year_data = df[df['Registered_Year'] == year]
    
    # Calculate counts and proportions
    pivot_data = year_data.groupby(['Parity', 'Place_of_Birth']).size().unstack(fill_value=0)
    pivot_data = pivot_data.reindex(parity_order)
    
    # Calculate row totals for labels
    row_totals = pivot_data.sum(axis=1)
    
    # Normalize to percentages
    pivot_pct = pivot_data.div(row_totals, axis=0) * 100
    
    # Plot Stacked Bar
    pivot_pct.plot(kind='bar', stacked=True, ax=ax, color=['#2ca02c', '#d62728'], alpha=0.8)
    
    ax.set_title(f'Delivery Setting by Birth Order - {year}', fontsize=18, fontweight='bold')
    ax.set_ylabel('Percentage of Births (%)')
    ax.set_xlabel('Birth Order (Parity)')
    ax.set_ylim(0, 115) 
    ax.legend(title='Place of Birth', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.setp(ax.get_xticklabels(), rotation=0)

    # Add Total Counts at the top of each bar
    for idx, total in enumerate(row_totals):
        ax.text(idx, 102, f'Total:\n{int(total):,}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    # Add text labels for Percentages and Absolute Counts inside segments
    y_offset = pd.Series([0.0] * len(pivot_pct), index=pivot_pct.index)
    for col in pivot_pct.columns:
        for idx in pivot_pct.index:
            val = pivot_pct.loc[idx, col]
            count = pivot_data.loc[idx, col]
            # find the integer position of idx for plotting
            pos = list(pivot_pct.index).index(idx)
            if val > 5: # Only label if segment is large enough
                ax.text(pos, y_offset[idx] + val/2, 
                        f'{val:.1f}%\n({int(count):,})', 
                        ha='center', va='center',
                        fontsize=8, color='white' if col == 'Non-Hospital' else 'black', fontweight='bold')
            y_offset[idx] += val

plt.tight_layout()
output_img = 'parity_by_hospital_distribution.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')

# Copy to results folder
if os.path.exists('Chapter_4_Results'):
    shutil.copy(output_img, 'Chapter_4_Results/' + output_img)

print(f"Visualization saved to {output_img} and synchronized with Chapter_4_Results/")
