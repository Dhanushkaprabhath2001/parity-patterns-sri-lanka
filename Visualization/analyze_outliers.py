import pandas as pd
import numpy as np

# 1. Load Data
df = pd.read_csv('merged_births_2000_2020.csv')
age_col = 'Age of Mother'

# --- Method A: Standard Statistical Outlier Analysis (IQR) ---
Q1 = df[age_col].quantile(0.25)
Q3 = df[age_col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

iqr_outliers = df[(df[age_col] < lower_bound) | (df[age_col] > upper_bound)]

# --- Method B: Study-Specific Outlier Analysis (Methodology) ---
# Ranges: <10, 10-14, 15-49, 50-60, >60
study_under_10 = df[df[age_col] < 10]
study_10_14 = df[(df[age_col] >= 10) & (df[age_col] < 15)]
study_15_49 = df[(df[age_col] >= 15) & (df[age_col] <= 49)]
study_50_60 = df[(df[age_col] > 49) & (df[age_col] <= 60)]
study_above_60 = df[df[age_col] > 60]

# --- Print Comparison Report ---
print("=== OUTLIER ANALYSIS: AGE OF MOTHER ===")
print(f"Total Records: {len(df)}")
print(f"\n1. STANDARD STATISTICAL (IQR) ANALYSIS:")
print(f"   Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"   Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"   Number of Statistical Outliers: {len(iqr_outliers)}")
print(f"   Percentage: {(len(iqr_outliers)/len(df)*100):.4f}%")

print(f"\n2. STUDY-SPECIFIC METHODOLOGY ANALYSIS:")
print(f"   < 10 (Excluded): {len(study_under_10)}")
print(f"   10-14 (Retained/Flagged): {len(study_10_14)}")
print(f"   15-49 (Standard Window): {len(study_15_49)}")
print(f"   50-60 (Retained/Flagged): {len(study_50_60)}")
print(f"   > 60 (Excluded): {len(study_above_60)}")

print(f"\n3. KEY DIFFERENCES:")
print(f"   Statistical outliers include ages between {upper_bound:.2f} and 60,")
print(f"   which the study methodology considers 'biologically possible' and retains.")

# Export stats for the final report
stats_df = pd.DataFrame({
    'Metric': ['Total Records', 'IQR Lower Bound', 'IQR Upper Bound', 'IQR Outlier Count', 
               'Study <10', 'Study 10-14', 'Study 15-49', 'Study 50-60', 'Study >60'],
    'Value': [len(df), lower_bound, upper_bound, len(iqr_outliers), 
              len(study_under_10), len(study_10_14), len(study_15_49), len(study_50_60), len(study_above_60)]
})
stats_df.to_csv('maternal_age_outlier_stats.csv', index=False)
