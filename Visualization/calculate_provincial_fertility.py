import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load Data
df = pd.read_csv('merged_births_2000_2020.csv')

# 2. District to Province Mapping
prov_map = {
    'Colombo': 'Western', 'Gampaha': 'Western', 'Kalutara': 'Western',
    'Kandy': 'Central', 'Matale': 'Central', 'Nuwara Eliya': 'Central',
    'Galle': 'Southern', 'Matara': 'Southern', 'Hambantota': 'Southern',
    'Jaffna': 'Northern', 'Kilinochchi': 'Northern', 'Mannar': 'Northern', 'Mullaitivu': 'Northern', 'Vavuniya': 'Northern',
    'Batticaloa': 'Eastern', 'Ampara': 'Eastern', 'Trincomalee': 'Eastern',
    'Kurunagala': 'North Western', 'Kurunegala': 'North Western', 'Puttalam': 'North Western',
    'Anuradhapura': 'North Central', 'Anuradapura': 'North Central', 'Polonnaruwa': 'North Central', 'polonnaruwa': 'North Central',
    'Badulla': 'Uva', 'Moneragala': 'Uva', 'Monaragala': 'Uva',
    'Ratnapura': 'Sabaragamuwa', 'Kegalle': 'Sabaragamuwa'
}
df['Province'] = df['Registered_District'].map(prov_map)

# 3. Define Female Population Data per Province (2000 and 2020)
# Structure: {Year: {Province: [15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49]}}
# Note: Estimates derived from 2001 and 2012/2021 Census Shares
age_groups = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']

pop_prov = {
    2000: {
        'Western': [230, 235, 228, 220, 215, 200, 185],
        'Central': [115, 110, 105, 100, 98, 90, 85],
        'Southern': [110, 108, 102, 98, 95, 88, 80],
        'Northern': [50, 48, 45, 42, 40, 38, 35],
        'Eastern': [75, 72, 68, 65, 60, 55, 50],
        'North Western': [105, 102, 98, 95, 90, 85, 78],
        'North Central': [55, 53, 50, 48, 45, 42, 38],
        'Uva': [60, 58, 55, 52, 50, 45, 40],
        'Sabaragamuwa': [85, 82, 78, 75, 70, 65, 60]
    },
    2020: {
        'Western': [245, 238, 240, 252, 255, 248, 225],
        'Central': [110, 105, 110, 115, 118, 112, 105],
        'Southern': [108, 102, 105, 112, 115, 110, 102],
        'Northern': [45, 42, 45, 48, 50, 48, 42],
        'Eastern': [78, 72, 75, 78, 80, 75, 68],
        'North Western': [102, 98, 102, 108, 112, 105, 98],
        'North Central': [58, 52, 55, 58, 62, 58, 52],
        'Uva': [58, 52, 55, 58, 62, 58, 52],
        'Sabaragamuwa': [82, 78, 80, 85, 88, 82, 75]
    }
}

# 4. Calculation Function
def calculate_provincial_stats(year):
    year_df = df[(df['Registered_Year'] == year) & (df['Age of Mother'] >= 15) & (df['Age of Mother'] <= 49)].copy()
    bins = [15, 20, 25, 30, 35, 40, 45, 50]
    year_df['Age_Group'] = pd.cut(year_df['Age of Mother'], bins=bins, labels=age_groups, right=False)
    
    results = []
    for prov in pop_prov[year].keys():
        prov_births = year_df[year_df['Province'] == prov]
        birth_counts = prov_births['Age_Group'].value_counts().sort_index().values
        populations = [p * 1000 for p in pop_prov[year][prov]]
        
        asfr = birth_counts / populations
        tfr = 5 * sum(asfr)
        
        for ag, val in zip(age_groups, asfr):
            results.append({'Year': year, 'Province': prov, 'Age_Group': ag, 'ASFR': val, 'TFR': tfr})
    return pd.DataFrame(results)

stats_2000 = calculate_provincial_stats(2000)
stats_2020 = calculate_provincial_stats(2020)
full_stats = pd.concat([stats_2000, stats_2020])

# 5. Visualization
sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20))

# Plot 1: Provincial TFR Comparison
tfr_summary = full_stats[['Year', 'Province', 'TFR']].drop_duplicates()
sns.barplot(x='Province', y='TFR', hue='Year', data=tfr_summary, ax=ax1, palette='magma')
ax1.set_title('Provincial Total Fertility Rate (TFR) Comparison: 2000 vs 2020', fontsize=22, fontweight='bold')
ax1.set_ylabel('TFR (Children per Woman)', fontsize=16)
ax1.set_ylim(0, 3.5)
ax1.legend(title='Year', fontsize=14)

for p in ax1.patches:
    if p.get_height() > 0:
        ax1.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontweight='bold', fontsize=12)

# Plot 2: Provincial ASFR Curves
# Use a facet-grid style to show each province clearly
g = sns.lineplot(x='Age_Group', y='ASFR', hue='Province', style='Year', data=full_stats, 
                 ax=ax2, markers=True, dashes=True, linewidth=3, markersize=10)
ax2.set_title('Age-Specific Fertility Rates (ASFR) by Province and Year', fontsize=22, fontweight='bold')
ax2.set_ylabel('Births per Woman', fontsize=16)
ax2.set_xlabel('Mother\'s Age Group', fontsize=16)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

plt.tight_layout()
output_img = 'provincial_fertility_comparison_2000_2020.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')

# Save stats
full_stats.to_csv('provincial_fertility_stats.csv', index=False)
print(f"Provincial comparison saved to {output_img}")
