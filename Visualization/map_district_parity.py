import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

# 1. Load Data
df = pd.read_csv('merged_births_2000_2020.csv')
gdf = gpd.read_file('shp/LKA_adm1.shp')

# 2. Standardize District Names in CSV
district_mapping = {
    'Anuradapura': 'Anuradhapura',
    'Kurunagala': 'Kurunegala',
    'polonnaruwa': 'Polonnaruwa',
    'Anuradhapura': 'Anuradhapura',
    'Polonnaruwa': 'Polonnaruwa',
    'Nuwara Eliya': 'Nuwara Eliya'
}
df['Registered_District'] = df['Registered_District'].str.strip().replace(district_mapping).str.title()

# 3. Standardize Parity for concise grouping
# Groups: 1st, 2nd, 3rd+
parity_map = {
    'First': '1st', 'Second': '2nd', 
    'Third': '3rd+', 'Fourth': '3rd+', 'Fifth': '3rd+', 
    'Sixth': '3rd+', 'Seventh': '3rd+', 'Eighth': '3rd+', 'Nineth': '3rd+'
}
df['Parity_Group'] = df['Birth_Order'].map(parity_map)

# 4. Aggregate Data: Counts and Percentages per District per Year
years = sorted(df['Registered_Year'].unique())
parity_order = ['1st', '2nd', '3rd+']

# Prepare the plot
fig, axes = plt.subplots(1, len(years), figsize=(30, 15))
colors = ['#440154', '#21918c', '#fde725'] # Viridis-like high contrast

for i, year in enumerate(years):
    ax = axes[i]
    year_df = df[df['Registered_Year'] == year]
    
    # Calculate Total Births per District
    dist_totals = year_df['Registered_District'].value_counts()
    
    # Calculate Parity Distribution per District
    dist_parity = year_df.groupby(['Registered_District', 'Parity_Group']).size().unstack(fill_value=0)
    dist_parity = dist_parity[parity_order] # Ensure order
    
    # Merge with Shapefile
    merged = gdf.merge(dist_totals.rename('Total_Births'), left_on='NAME_1', right_index=True, how='left')
    merged['Total_Births'] = merged['Total_Births'].fillna(0)
    
    # Background Map: Color by Total Births
    merged.plot(column='Total_Births', ax=ax, cmap='Greys', edgecolor='0.8', linewidth=0.5, alpha=0.3)
    
    # Calculate total for the whole year
    grand_total = int(dist_totals.sum())
    ax.set_title(f'Birth Parity by District - {year}\nTotal Births: {grand_total:,}', fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')

    # Add Labels and Small Stacked Bars/Pies for each district
    for idx, row in merged.iterrows():
        if row['Total_Births'] == 0: continue
        
        centroid = row['geometry'].centroid
        dist_name = row['NAME_1']
        total = int(row['Total_Births'])
        
        # Get parity data for this district
        p_data = dist_parity.loc[dist_name]
        p_pct = (p_data / total * 100).round(1)
        
        # Label: Name and Count
        label = f"{dist_name}\n{total:,}"
        ax.text(centroid.x, centroid.y + 0.1, label, ha='center', fontsize=8, fontweight='bold')
        
        # Draw a tiny custom stacked bar or indicator for parity
        # We'll just put the percentages in text to keep it clean but clear
        pct_text = f"1:{p_pct['1st']}%\n2:{p_pct['2nd']}%\n3+:{p_pct['3rd+']}%"
        ax.text(centroid.x, centroid.y - 0.15, pct_text, ha='center', fontsize=6, color='darkblue')

# Add a Global Legend for Parity Groups
legend_elements = [Patch(facecolor='darkblue', label='Parity Percentages displayed below District Name')]
fig.legend(handles=legend_elements, loc='lower center', fontsize=12)

plt.tight_layout()
output_img = 'district_parity_mapping_all_years.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Map visualization saved to {output_img}")
