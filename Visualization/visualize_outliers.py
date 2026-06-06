import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv('merged_births_2000_2020.csv')
age_col = 'Age of Mother'

# 2. Setup Visualization
plt.figure(figsize=(14, 10))
sns.set_style("white")

# Subplot 1: Boxplot for Statistical Outliers
plt.subplot(2, 1, 1)
sns.boxplot(x=df[age_col], color='skyblue', fliersize=5)
plt.title('Standard Statistical Outlier Analysis (Box Plot / IQR)', fontsize=16, fontweight='bold')
plt.xlabel('Maternal Age (Years)', fontsize=12)
# Annotate IQR bounds
Q1, Q3 = 24.0, 32.0
IQR = 8.0
lower_bound, upper_bound = 12.0, 44.0
plt.axvline(lower_bound, color='red', linestyle='--', label=f'Lower Bound ({lower_bound})')
plt.axvline(upper_bound, color='red', linestyle='--', label=f'Upper Bound ({upper_bound})')
plt.legend()

# Subplot 2: Methodology Visualization
plt.subplot(2, 1, 2)
# We'll plot a distribution but highlight the methodology zones
sns.histplot(df[age_col], bins=range(0, 100, 1), color='gray', alpha=0.3)
plt.yscale('log') # Log scale to see the rare cases (<15 and >50)

# Highlight Zones
plt.axvspan(0, 10, color='red', alpha=0.2, label='Impossible (Excluded <10)')
plt.axvspan(10, 15, color='orange', alpha=0.2, label='Rare: Adolescent (Retained 10-14)')
plt.axvspan(15, 49, color='green', alpha=0.1, label='Standard WHO (Retained 15-49)')
plt.axvspan(49, 60, color='orange', alpha=0.2, label='Rare: Perimenopausal (Retained 50-60)')
plt.axvspan(60, 100, color='red', alpha=0.2, label='Impossible (Excluded >60)')

plt.title('Study Methodology Outlier Analysis (Based on Biological Plausibility)', fontsize=16, fontweight='bold')
plt.xlabel('Maternal Age (Years)', fontsize=12)
plt.ylabel('Count (Log Scale)', fontsize=12)
plt.xlim(0, 70)
plt.legend(loc='upper right')

# Text annotation for the "999" outliers if they exist
outliers_999 = len(df[df[age_col] > 100])
if outliers_999 > 0:
    plt.annotate(f'{outliers_999} records at age 999\n(Excluded as Impossible)', 
                 xy=(65, 10), xytext=(55, 1000),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10, color='red', fontweight='bold')

plt.tight_layout()
output_img = 'maternal_age_outlier_comparison.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Comparison visualization saved to {output_img}")
