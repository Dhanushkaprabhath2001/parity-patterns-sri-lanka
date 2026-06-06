import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv('merged_births_2000_2020.csv')

# 2. Standardize Parity
parity_map = {
    'First': '1st', 'Second': '2nd', 'Third': '3rd', 
    'Fourth': '4th', 'Fifth': '5th', 'Sixth': '6th+', 
    'Seventh': '6th+', 'Eighth': '6th+', 'Nineth': '6th+'
}
df['Parity'] = df['Birth_Order'].map(parity_map)

# Define logical order for plotting
parity_order = ['1st', '2nd', '3rd', '4th', '5th', '6th+']
years = sorted(df['Registered_Year'].unique())

# 3. Visualization
# Set up a grid of plots - one for each year
fig, axes = plt.subplots(len(years), 1, figsize=(12, 20))
sns.set_style("whitegrid")

for i, year in enumerate(years):
    ax = axes[i]
    year_data = df[df['Registered_Year'] == year]
    
    # Filter for valid parity and sort
    plot_data = year_data.groupby(['Parity', 'Gender']).size().reset_index(name='Count')
    plot_data['Parity'] = pd.Categorical(plot_data['Parity'], categories=parity_order, ordered=True)
    plot_data = plot_data.sort_values('Parity')
    
    # Plot Grouped Bar Chart
    sns.barplot(x='Parity', y='Count', hue='Gender', data=plot_data, ax=ax, palette=['#FF69B4', '#1E90FF'])
    
    ax.set_title(f'Gender Distribution by Birth Order - {year}', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Births')
    ax.set_xlabel('Birth Order (Parity)')
    ax.legend(title='Gender', loc='upper right')

    # Add count labels on top of bars
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{int(p.get_height()):,}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points',
                        fontsize=9, fontweight='bold')

plt.tight_layout()
output_img = 'parity_gender_distribution_by_year.png'
plt.savefig(output_img, dpi=300)
print(f"Visualization saved to {output_img}")
