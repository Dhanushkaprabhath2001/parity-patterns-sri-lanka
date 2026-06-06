import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# 1. Load Data
df = pd.read_csv('merged_births_2000_2020.csv')

# 2. Map Categorical Birth Order to Numeric Values
order_map = {
    'First': 1, 'Second': 2, 'Third': 3, 'Fourth': 4, 
    'Fifth': 5, 'Sixth': 6, 'Seventh': 7, 'Eighth': 8, 'Nineth': 9
}
df['Parity_Num'] = df['Birth_Order'].map(order_map)
df = df.dropna(subset=['Parity_Num'])

years = sorted(df['Registered_Year'].unique())

# 3. Visualization
fig, axes = plt.subplots(len(years), 1, figsize=(10, 15), sharex=True)
sns.set_style("whitegrid")

skewness_results = {}

for i, year in enumerate(years):
    ax = axes[i]
    year_data = df[df['Registered_Year'] == year]['Parity_Num']
    
    # Calculate Frequency
    freq = year_data.value_counts().sort_index()
    
    # Calculate Skewness
    s_val = skew(year_data)
    skewness_results[year] = s_val
    
    # Plot Line Chart
    ax.plot(freq.index, freq.values, marker='o', linestyle='-', color='teal', linewidth=2)
    ax.fill_between(freq.index, freq.values, alpha=0.2, color='teal')
    
    ax.set_title(f'Birth Parity Distribution - {year} (Skewness: {s_val:.3f})', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Births')
    
    # Apply X-axis labels to ALL subplots as requested
    ax.set_xlabel('Birth Order', fontsize=12, fontweight='bold')
    ax.set_xticks(range(1, 10))
    ax.set_xticklabels(['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh', 'Eighth', 'Ninth'], fontsize=9)
    # Ensure tick labels are visible even with sharex=True
    ax.tick_params(labelbottom=True)
    
    # Label each point for clarity
    for x, y in zip(freq.index, freq.values):
        ax.annotate(f'{y:,}', xy=(x, y), xytext=(0, 5), textcoords="offset points", ha='center', fontsize=8)

# Adjust layout: increased hspace to 0.6 to prevent label/title overlap
plt.subplots_adjust(hspace=0.6, bottom=0.05, top=0.95)
output_img = 'parity_skewness_by_year.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Skewness charts saved to {output_img}")

# Print summary
print("\nSkewness Summary:")
for year, val in skewness_results.items():
    print(f"Year {year}: {val:.4f}")
