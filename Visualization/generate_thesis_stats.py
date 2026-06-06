import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load Raw Data (before any cleaning)
# Note: I'll reload from the merged CSV which already has some minimal cleaning,
# but for a true "exclusion log" I should really check the original Excels.
# However, I'll use the merged_births_2000_2020.csv as the base for this analysis.
df = pd.read_csv('merged_births_2000_2020.csv')
total_raw = len(df)

print(f"Initial Merge Count: {total_raw}")

# --- 4.1 Data Cleaning and Sample Derivation ---

# Step 1: Maternal Age Outliers (Based on study: <10 or >60)
# We actually retained 10-14 and 50-60 as rare.
exclude_age = df[(df['Age of Mother'] < 10) | (df['Age of Mother'] > 60)]
n_age = len(exclude_age)
df_clean = df.drop(exclude_age.index)

# Step 2: Birth Weight Outliers (Template mentions Step 2, but we don't have birth weight in 2000/2005)
# I'll check if birth weight exists
if 'Birth_Weight(grams)' in df.columns:
    exclude_bw = df_clean[(df_clean['Birth_Weight(grams)'] < 300) | (df_clean['Birth_Weight(grams)'] > 6000)]
    n_bw = len(exclude_bw)
    df_clean = df_clean.drop(exclude_bw.index)
else:
    n_bw = 0

# Steps 3-10: Listwise deletion for missing values
missing_log = {}
for col in ['Gender', 'Hospital or Not', 'Marital_Status', 'Race_of_Mother', 'Registered_District', 'Birth_Order', 'Age of Mother']:
    if col in df_clean.columns:
        n_missing = df_clean[col].isna().sum()
        missing_log[col] = n_missing
        df_clean = df_clean.dropna(subset=[col])

final_sample_n = len(df_clean)

print("\n--- TABLE 4.1: EXCLUSION LOG ---")
print(f"1. Maternal Age Outliers (<10 or >60): {n_age}")
print(f"2. Birth Weight Outliers (<300g or >6000g): {n_bw}")
for col, n in missing_log.items():
    print(f"Missing {col}: {n}")
print(f"Final Analytical Sample Size: {final_sample_n}")

# --- 4.2.1 Distribution of Birth Order ---
print("\n--- TABLE 4.3: PARITY DISTRIBUTION ---")
parity_dist = df_clean['Birth_Order'].value_counts().sort_index()
parity_pct = df_clean['Birth_Order'].value_counts(normalize=True).sort_index() * 100
parity_table = pd.DataFrame({'n': parity_dist, 'Percentage (%)': parity_pct.round(2)})
parity_table['Cumulative (%)'] = parity_table['Percentage (%)'].cumsum().round(2)
print(parity_table)

# --- 4.3 Bivariate Analysis (Chi-Square) ---
def calculate_cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

print("\n--- TABLE 4.4: CHI-SQUARE TEST RESULTS ---")
results_4_4 = []
for var in ['Gender', 'Hospital or Not', 'Marital_Status', 'Race_of_Mother', 'Registered_District']:
    contingency = pd.crosstab(df_clean[var], df_clean['Birth_Order'])
    chi2, p, dof, ex = chi2_contingency(contingency)
    v = calculate_cramers_v(contingency)
    results_4_4.append({'Variable': var, 'Chi2': chi2, 'p-value': p, 'df': dof, 'Cramers_V': v})

res_df = pd.DataFrame(results_4_4)
print(res_df)

# Save all results to a single summary file
with open('thesis_table_data.txt', 'w') as f:
    f.write("=== EXCLUSION LOG (Table 4.1) ===\n")
    f.write(f"Raw: {total_raw}\nAge Outliers: {n_age}\nBW Outliers: {n_bw}\n")
    for col, n in missing_log.items(): f.write(f"Missing {col}: {n}\n")
    f.write(f"Final N: {final_sample_n}\n\n")
    
    f.write("=== PARITY DISTRIBUTION (Table 4.3) ===\n")
    f.write(parity_table.to_string())
    f.write("\n\n")
    
    f.write("=== BIVARIATE RESULTS (Table 4.4) ===\n")
    f.write(res_df.to_string())

print("\nResults saved to thesis_table_data.txt")
