
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Setup paths (adjusting to the actual workspace)
# Based on the notebook, it expects a relative path
# DATA_PATH = Path("../../../data/no_missing") / "complete_birth_2020.xlsx"
# But we see the notebook in the current directory, and it seems to have run successfully.
# Let's try to load the file from where it is.
# The user's session context shows the file structure.

# Let's use the actual file path from the session context if possible, 
# or assume it's in a known relative location.
# In the notebook: DATA_PATH = Path("../../../data/no_missing")
# But I don't see that path in the directory listing.
# Wait, the user said "read the notebook", and the notebook HAS outputs.
# This means the notebook was executed.
# However, I need to run a script to get the EXPECTED counts.

# I will write a script that attempts to load the data, 
# but I need to make sure I have the right path.
# Looking at the directory structure provided in the prompt:
# E:\parity-patterns-sri-lanka-main\parity-patterns-sri-lanka\model_fitting\2020\patches 2020\
# contains models(2020)_revised.ipynb.
# It does NOT contain the raw excel file.

# Let's check if the raw file is in the parent directories.
# Or I can try to find where complete_birth_2020.xlsx is.

import os

def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
    return None

# Search for the data file starting from the project root if possible, 
# or just a few levels up.
data_file = find_file("complete_birth_2020.xlsx", "E:\\parity-patterns-sri-lanka-main")

if not data_file:
    print("Error: complete_birth_2020.xlsx not found.")
    exit(1)

print(f"Loading data from: {data_file}")
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

# --- CALCULATE EXPECTED COUNTS ---
PREDICTORS_CAT = ['age_group', 'marital_status', 'race_mother', 
                  'Gender ', 'Hospital or Not', 'multiple_birth', 'province_mother']

print("\n" + "="*80)
print(f"{'Variable':<20} | {'Min Expected':<15} | {'Cells < 5':<12} | {'Assumption Met'}")
print("-" * 80)

for var in PREDICTORS_CAT:
    ct = pd.crosstab(df['parity'], df[var])
    _, _, _, expected = stats.chi2_contingency(ct)
    
    min_exp = expected.min()
    cells_below_5 = (expected < 5).sum()
    total_cells = expected.size
    pct_below_5 = (cells_below_5 / total_cells) * 100
    
    # Assumption: No cell < 1 and < 20% of cells < 5
    met = "YES" if (min_exp >= 1 and pct_below_5 <= 20) else "NO"
    
    print(f"{var:<20} | {min_exp:<15.4f} | {cells_below_5:<4} ({pct_below_5:>5.1f}%) | {met}")

print("="*80)
