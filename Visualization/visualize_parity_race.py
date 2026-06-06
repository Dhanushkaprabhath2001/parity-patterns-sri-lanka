import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv('merged_births_2000_2020.csv')

# 2. Standardize Parity
parity_map = {
    'First': '1st', 'Second': '2nd', 'Third': '3rd', 
    'Fourth': '4th', 'Fifth': '5th', 'Sixth': '6th+', 
    'Seventh': '6th+', 'Eighth': '6th+', 'Nineth': '6th+'
}
df['Parity'] = df['Birth_Order'].map(parity_map)

# 3. Group smaller racial categories into 'Others' for cleaner visualization
main_races = ['Sinhalese', 'Srilankan Tamil', 'Srilankan Moor', 'Indian Tamil']
df['Race_Group'] = df['Race_of_Mother'].apply(lambda x: x if x in main_races else 'Others')

# Define order for plotting
parity_order = ['1st', '2nd', '3rd', '4th', '5th', '6th+']
years = sorted(df['Registered_Year'].unique())

# 4. Visualization
# We'll create a stacked percentage bar chart for each year to compare proportions
fig, axes = plt.subplots(len(years), 1, figsize=(14, 25))
sns.set_style("whitegrid")

for i, year in enumerate(years):
    ax = axes[i]
    year_data = df[df['Registered_Year'] == year]
    
    # Calculate counts and percentages
    race_parity = year_data.groupby(['Race_Group', 'Parity']).size().unstack(fill_value=0)
    existing_cols = [p for p in parity_order if p in race_parity.columns]
    race_parity = race_parity[existing_cols]
    
    # Calculate row totals and SORT DESCENDING
    row_totals = race_parity.sum(axis=1).sort_values(ascending=False)
    # Reindex race_parity to match sorted order
    race_parity = race_parity.reindex(row_totals.index)
    
    # Normalize to percentages for the plot
    race_parity_pct = race_parity.div(row_totals, axis=0) * 100
    
    # Plot Stacked Bar (still using percentages for visual structure)
    race_parity_pct.plot(kind='bar', stacked=True, ax=ax, colormap='Spectral')
    
    ax.set_title(f'Birth Parity Distribution by Mother\'s Race - {year} (Sorted by Total Births)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Percentage of Births (%)')
    ax.set_xlabel('Race of Mother (Sorted Descending by Count)')
    ax.set_ylim(0, 115) 
    ax.legend(title='Parity', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.setp(ax.get_xticklabels(), rotation=0)

    # Add Total Counts at the top of each bar
    for idx, (race, total) in enumerate(row_totals.items()):
        ax.text(idx, 102, f'Total:\n{int(total):,}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkred')

    # Add text labels for Percentages and Absolute Counts inside segments
    y_offset = pd.Series([0.0] * len(race_parity_pct), index=race_parity_pct.index)
    for col in race_parity_pct.columns:
        for idx in race_parity_pct.index:
            val = race_parity_pct.loc[idx, col]
            count = race_parity.loc[idx, col]
            if val > 6: # Only label if segment is large enough
                # find the integer position of idx for plotting
                pos = list(race_parity_pct.index).index(idx)
                ax.text(pos, y_offset[idx] + val/2, 
                        f'{val:.1f}%\n({int(count):,})', 
                        ha='center', va='center',
                        fontsize=8, color='black', fontweight='bold')
            y_offset[idx] += val

plt.tight_layout()
output_img = 'parity_by_race_distribution_with_counts.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Visualization saved to {output_img}")

# Also save the percentage data for reference
race_parity_summary = df.groupby(['Registered_Year', 'Race_Group', 'Parity']).size().unstack(fill_value=0)
race_parity_summary = race_parity_summary.div(race_parity_summary.sum(axis=1), axis=0) * 100
race_parity_summary.to_csv('parity_race_percentages_by_year.csv')
print("Percentage data saved to parity_race_percentages_by_year.csv")
