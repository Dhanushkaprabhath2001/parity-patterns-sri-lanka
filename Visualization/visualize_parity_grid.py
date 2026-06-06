import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os

# 1. Load Data
df = pd.read_csv('merged_births_2000_2020.csv')
gdf = gpd.read_file('shp/LKA_adm1.shp')

# 2. Standardize District Names
district_mapping = {
    'Anuradapura': 'Anuradhapura',
    'Kurunagala': 'Kurunegala',
    'polonnaruwa': 'Polonnaruwa',
    'Anuradhapura': 'Anuradhapura',
    'Polonnaruwa': 'Polonnaruwa',
    'Nuwara Eliya': 'Nuwara Eliya'
}
df['Registered_District'] = df['Registered_District'].str.strip().replace(district_mapping).str.title()

# 3. Standardize Parity Groups
parity_map = {
    'First': '1st', 'Second': '2nd', 
    'Third': '3rd+', 'Fourth': '3rd+', 'Fifth': '3rd+', 
    'Sixth': '3rd+', 'Seventh': '3rd+', 'Eighth': '3rd+', 'Nineth': '3rd+'
}
df['Parity_Group'] = df['Birth_Order'].map(parity_map)

# 4. Process Data for Grid
years = sorted(df['Registered_Year'].unique())
parity_groups = ['1st', '2nd', '3rd+']

# Prepare the Grid Plot (3 Rows for Parity, 5 Columns for Years)
fig, axes = plt.subplots(len(parity_groups), len(years), figsize=(30, 20))

for r, parity in enumerate(parity_groups):
    # Calculate global min/max for this parity row to have a consistent color scale
    # This is important so colors are comparable across years for the same parity
    yearly_dist_parity = df.groupby(['Registered_Year', 'Registered_District', 'Parity_Group']).size().unstack(fill_value=0)
    yearly_dist_totals = df.groupby(['Registered_Year', 'Registered_District']).size()
    
    # Calculate percentage for this specific parity group across all year/district combos
    all_pcts = (yearly_dist_parity[parity] / yearly_dist_totals * 100).dropna()
    vmin, vmax = all_pcts.min(), all_pcts.max()

    for c, year in enumerate(years):
        ax = axes[r, c]
        year_df = df[df['Registered_Year'] == year]
        
        # Calculate specific parity percentage for this year/district
        dist_totals = year_df['Registered_District'].value_counts()
        dist_parity_counts = year_df[year_df['Parity_Group'] == parity]['Registered_District'].value_counts()
        dist_pcts = (dist_parity_counts / dist_totals * 100).fillna(0)
        
        # Merge with Shapefile
        merged = gdf.merge(dist_pcts.rename('Percentage'), left_on='NAME_1', right_index=True, how='left')
        merged['Percentage'] = merged['Percentage'].fillna(0)
        
        # Plot Map
        merged.plot(column='Percentage', ax=ax, cmap='YlOrRd', vmin=vmin, vmax=vmax, edgecolor='0.4', linewidth=0.3)
        
        # Titles only on the top row and left column for cleanliness
        if r == 0:
            ax.set_title(f'Year: {year}', fontsize=24, fontweight='bold', pad=15)
        if c == 0:
            ax.set_ylabel(f'Parity: {parity}', fontsize=24, fontweight='bold', labelpad=20)
            # Remove the default axis to keep it clean but keep the ylabel
            ax.set_yticks([])
            ax.set_xticks([])
        else:
            ax.axis('off')
            
        # Add District Name and Pct Label
        for idx, row in merged.iterrows():
            centroid = row['geometry'].centroid
            label = f"{row['NAME_1']}\n{row['Percentage']:.1f}%"
            ax.text(centroid.x, centroid.y, label, ha='center', fontsize=7, fontweight='bold', color='black')

# Add colorbars for each row to show the scale
for r in range(len(parity_groups)):
    # Create a dummy mappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=all_pcts.min(), vmax=all_pcts.max()))
    sm._A = []
    cbar_ax = fig.add_axes([0.92, 0.7 - r*0.25, 0.015, 0.2]) # Custom position for each row's colorbar
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f'% of {parity_groups[r]} Births', fontsize=12)

plt.subplots_adjust(left=0.05, right=0.9, top=0.92, bottom=0.05, wspace=0.1, hspace=0.1)
output_img = 'parity_grid_maps_by_order_and_year.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Grid map saved to {output_img}")
