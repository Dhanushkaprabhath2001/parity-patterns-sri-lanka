import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os

# 1. Load Data
df = pd.read_csv('merged_births_2000_2020.csv')
gdf = gpd.read_file('shp/LKA_adm1.shp')

# 2. Standardize District Names in CSV
# Mapping from CSV unique values to Shapefile NAME_1 values
district_mapping = {
    'Anuradapura': 'Anuradhapura',
    'Kurunagala': 'Kurunegala',
    'polonnaruwa': 'Polonnaruwa',
    'Anuradhapura': 'Anuradhapura',
    'Polonnaruwa': 'Polonnaruwa',
    'Nuwara Eliya': 'Nuwara Eliya'
}

# Apply mapping and basic cleaning
df['Registered_District'] = df['Registered_District'].str.strip()
df['Registered_District'] = df['Registered_District'].replace(district_mapping)
# Ensure casing matches shapefile (Title Case)
df['Registered_District'] = df['Registered_District'].str.title()
# Fix cases where Title() might break multi-word names (none expected here based on shapefile check)

# 3. Aggregate Birth Counts
agg_df = df.groupby(['Registered_Year', 'Registered_District']).size().reset_index(name='Birth_Count')

# 4. Prepare Visualization
years = sorted(agg_df['Registered_Year'].unique())
fig, axes = plt.subplots(1, len(years), figsize=(25, 12), sharex=True, sharey=True)

# Find global min/max for consistent color scale (One Key)
vmin = agg_df['Birth_Count'].min()
vmax = agg_df['Birth_Count'].max()

for i, year in enumerate(years):
    ax = axes[i]
    year_data = agg_df[agg_df['Registered_Year'] == year]
    total_year_births = year_data['Birth_Count'].sum()
    
    # Merge with shapefile
    merged = gdf.merge(year_data, left_on='NAME_1', right_on='Registered_District', how='left')
    merged['Birth_Count'] = merged['Birth_Count'].fillna(0)
    
    # Plot Map
    merged.plot(column='Birth_Count', ax=ax, cmap='YlOrRd', legend=False, 
                vmin=vmin, vmax=vmax, edgecolor='0.5', linewidth=0.5)
    
    ax.set_title(f'Registered Births - {year}\n(Total: {total_year_births:,})', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add Labels (District Name and Count)
    for idx, row in merged.iterrows():
        # Get centroid for label placement
        centroid = row['geometry'].centroid
        # Annotate with Name and Count
        label = f"{row['NAME_1']}\n{int(row['Birth_Count']):,}"
        ax.annotate(label, xy=(centroid.x, centroid.y), xytext=(0, 0), 
                    textcoords="offset points", ha='center', fontsize=6, 
                    color='black', fontweight='bold', alpha=0.8)

# Add a single colorbar for the whole figure
sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
# Increased pad and moved to bottom with more room
cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
cbar.set_label('Number of Registered Births', fontsize=14)

# Adjust spacing manually to avoid overlap and warning
plt.subplots_adjust(bottom=0.2, top=0.9, wspace=0.1)
output_img = 'yearly_birth_maps_sri_lanka.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Maps saved to {output_img}")

# Also save the aggregated data for reference
agg_df.to_csv('aggregated_births_by_district_year.csv', index=False)
print("Aggregated data saved to aggregated_births_by_district_year.csv")
