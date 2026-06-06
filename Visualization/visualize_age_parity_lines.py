import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

# 1. Load Data
df = pd.read_csv('merged_births_2000_2020.csv')

# 2. Data Cleaning
# Filter maternal age to realistic range (10 to 60)
df = df[(df['Age of Mother'] >= 10) & (df['Age of Mother'] <= 60)]

# Standardize Parity
parity_map = {
    'First': '1st', 'Second': '2nd', 'Third': '3rd', 
    'Fourth': '4th', 'Fifth': '5th', 'Sixth': '6th+', 
    'Seventh': '6th+', 'Eighth': '6th+', 'Nineth': '6th+'
}
df['Parity'] = df['Birth_Order'].map(parity_map)
parity_order = ['1st', '2nd', '3rd', '4th', '5th', '6th+']
years = sorted(df['Registered_Year'].unique())

# 3. Visualization
# Set up a grid - one chart per year
fig, axes = plt.subplots(len(years), 1, figsize=(14, 25), sharex=True)
sns.set_style("whitegrid")

for i, year in enumerate(years):
    ax = axes[i]
    year_data = df[df['Registered_Year'] == year]
    
    # Plot smoothed density (KDE) with increased bandwidth adjustment for smoothness
    sns.kdeplot(data=year_data, x='Age of Mother', hue='Parity', 
                hue_order=parity_order, palette='viridis', 
                linewidth=3, ax=ax, common_norm=False, legend=True, bw_adjust=1.5)
    
    ax.set_title(f'Maternal Age Distribution by Parity - {year}', fontsize=22, fontweight='bold', pad=20)
    ax.set_ylabel('Density', fontsize=16, fontweight='bold')
    ax.set_xlabel('Maternal Age (Years)', fontsize=16, fontweight='bold')
    ax.set_xlim(12, 55)
    
    # Reposition the existing legend
    leg = ax.get_legend()
    if leg:
        leg.set_title('Birth Order')
        plt.setp(leg.get_title(), fontsize=14, fontweight='bold')
    
    # Ensure X-axis ticks and labels are visible on ALL subplots
    ax.tick_params(labelbottom=True)

# Adjust layout to prevent overlap with legends and titles
plt.subplots_adjust(hspace=0.6, bottom=0.05, top=0.95)
output_img = 'age_parity_line_plots_by_year.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')

# Copy to results folder
if os.path.exists('Chapter_4_Results'):
    shutil.copy(output_img, 'Chapter_4_Results/' + output_img)

print(f"Visualization saved to {output_img} and synchronized with Chapter_4_Results/")
