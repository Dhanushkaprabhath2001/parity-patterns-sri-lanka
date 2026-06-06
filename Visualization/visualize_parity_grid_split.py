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

# 3. Standardize Parity into individual categories (No 3rd+ grouping)
parity_map = {
    'First': '1st', 'Second': '2nd', 'Third': '3rd', 
    'Fourth': '4th', 'Fifth': '5th', 
    'Sixth': '6th+', 'Seventh': '6th+', 'Eighth': '6th+', 'Nineth': '6th+'
}
df['Parity_Group'] = df['Birth_Order'].map(parity_map)

years = sorted(df['Registered_Year'].unique())

def generate_grid_image(parity_list, filename):
    fig, axes = plt.subplots(len(parity_list), len(years), figsize=(32, 22))
    cmap = 'YlOrRd'
    
    # Calculate all percentages for THIS image's parity list to find a GLOBAL scale
    yearly_dist_parity = df.groupby(['Registered_Year', 'Registered_District', 'Parity_Group']).size().unstack(fill_value=0)
    yearly_dist_totals = df.groupby(['Registered_Year', 'Registered_District']).size()
    
    # Filter for only the parities in this image
    relevant_pcts = []
    for p in parity_list:
        p_pcts = (yearly_dist_parity[p] / yearly_dist_totals * 100).dropna()
        relevant_pcts.extend(p_pcts.tolist())
    
    global_vmin = min(relevant_pcts)
    global_vmax = max(relevant_pcts)

    for r, parity in enumerate(parity_list):
        for c, year in enumerate(years):
            ax = axes[r, c]
            year_df = df[df['Registered_Year'] == year]
            
            # Calculate percentages
            dist_totals = year_df['Registered_District'].value_counts()
            dist_parity_counts = year_df[year_df['Parity_Group'] == parity]['Registered_District'].value_counts()
            dist_pcts = (dist_parity_counts / dist_totals * 100).fillna(0)
            
            merged = gdf.merge(dist_pcts.rename('Percentage'), left_on='NAME_1', right_index=True, how='left')
            merged['Percentage'] = merged['Percentage'].fillna(0)
            
            # Plot with global scale
            merged.plot(column='Percentage', ax=ax, cmap=cmap, vmin=global_vmin, vmax=global_vmax, 
                        edgecolor='0.4', linewidth=0.2)
            
            # Remove all axes boxes/lines
            ax.axis('off')
            
            # Top row titles (Year)
            if r == 0:
                ax.set_title(f'Year: {year}', fontsize=28, fontweight='bold', pad=10)
            
            # Left column labels (Parity)
            if c == 0:
                # Place text manually to avoid axis boxes
                ax.text(-0.2, 0.5, f'Parity: {parity}', transform=ax.transAxes, 
                        fontsize=28, fontweight='bold', va='center', ha='right', rotation=90)
            
            # District Labels
            for idx, row in merged.iterrows():
                centroid = row['geometry'].centroid
                label = f"{row['NAME_1']}\n{row['Percentage']:.1f}%"
                ax.text(centroid.x, centroid.y, label, ha='center', fontsize=7, fontweight='bold')

    # Add ONE Global Colorbar for the whole PNG
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=global_vmin, vmax=global_vmax))
    sm._A = []
    # Position colorbar at the bottom
    cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02]) 
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'Percentage (%) of Births for {parity_list[0]} to {parity_list[-1]}', fontsize=16, fontweight='bold')

    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.1, wspace=0.02, hspace=0.05)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Clean Grid Map saved to {filename}")

# Generate Part 1: 1st, 2nd, 3rd (Global scale for low parity)
generate_grid_image(['1st', '2nd', '3rd'], 'parity_grid_part1.png')

# Generate Part 2: 4th, 5th, 6th+ (Global scale for high parity)
generate_grid_image(['4th', '5th', '6th+'], 'parity_grid_part2.png')
