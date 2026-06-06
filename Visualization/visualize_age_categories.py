import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv('merged_births_2000_2020.csv')

# 2. Data Cleaning
# Filter maternal age to a realistic range (12 to 55)
df = df[(df['Age of Mother'] >= 12) & (df['Age of Mother'] <= 55)]

# 3. Categorize Age of Mother
bins = [0, 20, 25, 30, 35, 40, 100]
labels = ['<20', '20-24', '25-29', '30-34', '35-39', '40+']
df['Age_Category'] = pd.cut(df['Age of Mother'], bins=bins, labels=labels, right=False)

# 4. Generate Frequency Table
years = sorted(df['Registered_Year'].unique())
age_dist_counts = df.groupby(['Registered_Year', 'Age_Category']).size().unstack(fill_value=0)
age_dist_pct = age_dist_counts.div(age_dist_counts.sum(axis=1), axis=0) * 100

# Save table to CSV
age_dist_report = age_dist_counts.copy()
for col in labels:
    age_dist_report[f'{col} (%)'] = age_dist_pct[col].round(2)

age_dist_report.to_csv('maternal_age_categories_distribution.csv')
print("Frequency table saved to maternal_age_categories_distribution.csv")

# 5. Visualization
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# Stacked bar chart for proportions
ax = age_dist_pct.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='RdYlBu_r')

plt.title('Maternal Age Composition by Year (2000-2020)', fontsize=16, fontweight='bold')
plt.ylabel('Percentage of Total Births (%)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Age Category', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add percentage labels
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    if height > 3: # Only label if segment is large enough
        ax.text(x + width/2, 
                y + height/2, 
                f'{height:.1f}%', 
                ha='center', va='center',
                fontsize=9, color='black', fontweight='bold')

plt.tight_layout()
output_img = 'maternal_age_composition_by_year.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Visualization saved to {output_img}")
