
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import os

def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
    return None

data_file = find_file("complete_birth_2020.xlsx", "E:\\parity-patterns-sri-lanka-main")
df_raw = pd.read_excel(data_file)

# --- REPRODUCE CLEANING FROM NOTEBOOK ---
df = df_raw.copy()
df = df.rename(columns={
    'Birth_Order'          : 'parity',
    'Age of Mother'        : 'maternal_age',
    'Marital_Status'       : 'marital_status',
    'Race_of_Mother'       : 'race_mother',
    'Gender'               : 'Gender ',
    'Hospital or Not'      : 'Hospital or Not',
    'Multiple_Birth_Status': 'multiple_birth',
    'Registered_District'  : 'reg_district',
    'District_of_Mother'   : 'district_mother',
})

BIRTH_ORDER_MAP = {
    'First'  : 1, 'Second' : 2, 'Third'  : 3, 'Fourth' : 4,
    'Fifth'  : 5, 'Sixth'  : 6, 'Seventh': 7, 'Eighth' : 8, 'Nineth' : 9,
}
df['parity'] = df['parity'].astype(str).str.strip().map(BIRTH_ORDER_MAP)

# Collapse parity: 1, 2, 3, 4+
df['parity_collapsed'] = df['parity'].clip(upper=4).astype(int)

DISTRICT_PROVINCE_MAP = {
    'Colombo': 'Western', 'Gampaha': 'Western', 'Kalutara': 'Western',
    'Kandy': 'Central', 'Matale': 'Central', 'Nuwara Eliya': 'Central',
    'Galle': 'Southern', 'Matara': 'Southern', 'Hambantota': 'Southern',
    'Jaffna': 'Northern', 'Kilinochchi': 'Northern', 'Mannar': 'Northern', 'Vavuniya': 'Northern', 'Mullaitivu': 'Northern',
    'Batticaloa': 'Eastern', 'Ampara': 'Eastern', 'Trincomalee': 'Eastern',
    'Kurunegala': 'North Western', 'Puttalam': 'North Western',
    'Anuradhapura': 'North Central', 'Polonnaruwa': 'North Central',
    'Badulla': 'Uva', 'Moneragala': 'Uva',
    'Ratnapura': 'Sabaragamuwa', 'Kegalle': 'Sabaragamuwa'
}
df['province_mother'] = df['district_mother'].map(DISTRICT_PROVINCE_MAP)

df['maternal_age'] = pd.to_numeric(df['maternal_age'], errors='coerce')
age_bins   = [0, 19, 24, 29, 34, 120]
age_labels = ['<20', '20-24', '25-29', '30-34', '35+']
df['age_group'] = pd.cut(df['maternal_age'], bins=age_bins, labels=age_labels)

CAT_COLS = ['Gender ', 'Hospital or Not', 'multiple_birth', 'marital_status', 'race_mother']
for col in CAT_COLS:
    df[col] = df[col].astype(str).str.strip().str.title()

df = df[(df['maternal_age'] >= 10) & (df['maternal_age'] <= 60)]
df.dropna(subset=['parity', 'age_group', 'race_mother'], inplace=True)

# --- CALCULATE EXPECTED COUNTS FOR COLLAPSED PARITY ---
PREDICTORS_CAT = ['age_group', 'marital_status', 'race_mother', 
                  'Gender ', 'Hospital or Not', 'multiple_birth', 'province_mother']

print("\n" + "="*80)
print("EXPECTED COUNTS FOR COLLAPSED PARITY (1, 2, 3, 4+)")
print("="*80)
print(f"{'Variable':<20} | {'Min Expected':<15} | {'Cells < 5':<12} | {'Assumption Met'}")
print("-" * 80)

for var in PREDICTORS_CAT:
    ct = pd.crosstab(df['parity_collapsed'], df[var])
    _, _, _, expected = stats.chi2_contingency(ct)
    
    min_exp = expected.min()
    cells_below_5 = (expected < 5).sum()
    total_cells = expected.size
    pct_below_5 = (cells_below_5 / total_cells) * 100
    
    # Assumption: No cell < 1 and < 20% of cells < 5
    met = "YES" if (min_exp >= 1 and pct_below_5 <= 20) else "NO"
    
    print(f"{var:<20} | {min_exp:<15.4f} | {cells_below_5:<4} ({pct_below_5:>5.1f}%) | {met}")

print("="*80)
