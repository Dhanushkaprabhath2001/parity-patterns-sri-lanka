import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv('merged_births_2000_2020.csv')

# 2. Data Cleaning
# Filter maternal age to a realistic range (12 to 55)
df = df[(df['Age of Mother'] >= 12) & (df['Age of Mother'] <= 55)]

# 3. Visualization
sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))

# --- Plot 1: Overlaid Density Plots (Line Graph style) ---
# This shows the shift in the "Peak" childbearing age across years
sns.kdeplot(data=df, x='Age of Mother', hue='Registered_Year', 
            palette='viridis', common_norm=False, fill=True, alpha=0.1, linewidth=2.5, ax=ax1)

ax1.set_title('Maternal Age Distribution: Comparison Across Years (Density Curves)', fontsize=18, fontweight='bold')
ax1.set_xlabel('Maternal Age (Years)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Density', fontsize=14, fontweight='bold')
ax1.set_xlim(12, 55)

# --- Plot 2: Multi-panel Histograms ---
# This shows the raw frequency for each year separately
years = sorted(df['Registered_Year'].unique())
for i, year in enumerate(years):
    # Using a secondary plot to overlay histograms for comparison
    sns.histplot(df[df['Registered_Year'] == year]['Age of Mother'], 
                 bins=range(12, 56), label=str(year), color=sns.color_palette("viridis")[i],
                 element="step", fill=False, linewidth=2, ax=ax2)

ax2.set_title('Maternal Age Frequency Polygons (Step Histograms)', fontsize=18, fontweight='bold')
ax2.set_xlabel('Maternal Age (Years)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Number of Births', fontsize=14, fontweight='bold')
ax2.set_xlim(12, 55)
ax2.legend(title='Year')

plt.tight_layout()
output_img = 'maternal_age_distribution_curves.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Distribution curves saved to {output_img}")
