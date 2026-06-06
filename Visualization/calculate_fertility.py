import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Birth Data
df = pd.read_csv('merged_births_2000_2020.csv')

# 2. Define Female Population Data (in thousands, from UN/DCS estimates)
# Age groups: 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49
pop_data = {
    2000: [854, 842, 786, 742, 735, 682, 598],
    2005: [824, 856, 844, 788, 744, 737, 684],
    2010: [791, 826, 858, 846, 790, 746, 739],
    2015: [756, 793, 828, 860, 848, 792, 748],
    2020: [791, 758, 795, 830, 862, 850, 794]
}
age_groups = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']

# 3. Clean Birth Data (Filter valid reproductive ages 15-49)
df = df[(df['Age of Mother'] >= 15) & (df['Age of Mother'] <= 49)]

# Define bins for ASFR
bins = [15, 20, 25, 30, 35, 40, 45, 50]
labels = age_groups
df['Age_Group'] = pd.cut(df['Age of Mother'], bins=bins, labels=labels, right=False)

# 4. Calculate ASFR and TFR
tfr_results = []
asfr_all_years = []

for year in sorted(pop_data.keys()):
    year_births = df[df['Registered_Year'] == year]
    birth_counts = year_births['Age_Group'].value_counts().sort_index()
    
    # Populations are in thousands, so multiply by 1000
    populations = [p * 1000 for p in pop_data[year]]
    
    # ASFR = Births / Female Population in that age group
    asfr = (birth_counts.values / populations)
    
    # TFR = 5 * Sum(ASFR) 
    # (The factor of 5 is because these are 5-year age groups)
    tfr = 5 * sum(asfr)
    
    tfr_results.append({'Year': year, 'TFR': tfr})
    
    # Store ASFR for visualization
    for label, val in zip(labels, asfr):
        asfr_all_years.append({'Year': year, 'Age_Group': label, 'ASFR': val})

tfr_df = pd.DataFrame(tfr_results)
asfr_df = pd.DataFrame(asfr_all_years)

# 5. Visualizations
sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Plot TFR Trend
sns.barplot(x='Year', y='TFR', data=tfr_df, palette='Reds_d', ax=ax1)
ax1.set_title('Total Fertility Rate (TFR) by Year - Sri Lanka', fontsize=16, fontweight='bold')
ax1.set_ylabel('TFR (Children per Woman)')
ax1.set_ylim(0, 3) # TFR is usually between 1.5 and 2.5 for SL
for p in ax1.patches:
    ax1.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontweight='bold')

# Plot ASFR Curves
# Using 'rocket' palette for better contrast (replaces 'viridis' which had yellow)
sns.lineplot(x='Age_Group', y='ASFR', hue='Year', data=asfr_df, marker='o', ax=ax2, 
             palette='rocket', linewidth=3, markersize=8)
ax2.set_title('Age-Specific Fertility Rates (ASFR)', fontsize=16, fontweight='bold')
ax2.set_ylabel('Births per Woman in Age Group')
ax2.set_xlabel('Mother\'s Age Group')

plt.tight_layout()
output_img = 'fertility_analysis_sri_lanka.png'
plt.savefig(output_img, dpi=300)
# Also copy to the results folder
import shutil
shutil.copy(output_img, 'Chapter_4_Results/fertility_analysis_sri_lanka.png')
print(f"Updated fertility analysis saved to {output_img} and Chapter_4_Results/")

# Save TFR data to CSV
tfr_df.to_csv('total_fertility_rates.csv', index=False)
print("TFR data saved to total_fertility_rates.csv")
