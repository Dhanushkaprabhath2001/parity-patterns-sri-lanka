import nbformat

with open('descriptive_analysis_crvs_2000_2020.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Cell 4 is usually the one with DATA_PATHS and VARIABLE_REGISTRY based on previous read
# Let's find it by content
for cell in nb.cells:
    if 'DATA_PATHS' in cell.source:
        cell.source = """# ════════════════════════════════════════════════════════════════════════════════
#  FILE PATHS — update these to match your actual file locations
# ════════════════════════════════════════════════════════════════════════════════
DATA_PATHS = {
    2000: 'complete_birth_2000.xlsx',
    2005: 'complete_birth_2005.xlsx',
    2010: 'complete_birth_2010.xlsx',
    2015: 'complete_birth_2015.xlsx',
    2020: 'complete_birth_2020.xlsx',
}

STUDY_YEARS = [2000, 2005, 2010, 2015, 2020]

# ════════════════════════════════════════════════════════════════════════════════
#  WHO / IUPAC RANGE THRESHOLDS
# ════════════════════════════════════════════════════════════════════════════════

# Maternal age — valid range (completed years)
AGE_MIN, AGE_MAX = 10, 55           # outside → exclusion
AGE_FLAG_LOW  = 15                   # 10–14 → flagged adolescent
AGE_FLAG_HIGH = 49                   # 50–55 → flagged perimenopausal

# Birth weight — valid range (grams, SI unit)
BW_MIN, BW_MAX = 400, 6000          # outside → exclusion
BW_ELBW      = 1000                  # < 1000 g → ELBW
BW_VLBW      = 1500                  # < 1500 g → VLBW
BW_LBW       = 2500                  # < 2500 g → LBW (WHO threshold)
BW_NORMAL_HI = 4000                  # ≥ 4000 g → Macrosomia

# ════════════════════════════════════════════════════════════════════════════════
#  VARIABLE REGISTRY — which variables exist in which year
# ════════════════════════════════════════════════════════════════════════════════
VARIABLE_REGISTRY = {
    # Core variables — present in ALL years (2000–2020)
    'Birth_Order':           [2000, 2005, 2010, 2015, 2020],
    'Age_of_Mother':         [2000, 2005, 2010, 2015, 2020],
    'Marital_Status':        [2000, 2005, 2010, 2015, 2020],
    'Race_of_Mother':        [2000, 2005, 2010, 2015, 2020],
    'Gender':                [2000, 2005, 2010, 2015, 2020],
    'Hospital_or_Not':       [2000, 2005, 2010, 2015, 2020],
    'District_of_Mother':    [2000, 2005, 2010, 2015, 2020],
    'Registered_District':   [2000, 2005, 2010, 2015, 2020],

    # Extended variables — only in 2010+
    'Multiple_Birth_Status': [2010, 2015, 2020],
    'Birth_Weight_g':        [2010, 2015, 2020],
    'Race_of_Father':        [2010, 2015, 2020],
}

# ════════════════════════════════════════════════════════════════════════════════
#  PARITY / REPORTING CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════
PARITY_LABELS = {
    1: 'First',  2: 'Second', 3: 'Third',  4: 'Fourth',
    5: 'Fifth',  6: 'Sixth',  7: 'Seventh',8: 'Eighth',
    9: 'Ninth'
}

PARITY_MAP = {v: k for k, v in PARITY_LABELS.items()}

AGE_ORDER = ['<20', '20–24', '25–29', '30–34', '35+']

BW_CAT_ORDER = [
    'ELBW (<1000 g)', 'VLBW (1000–1499 g)', 'LBW (1500–2499 g)',
    'Normal (2500–3999 g)', 'Macrosomia (≥4000 g)'
]

print('Configuration loaded.')
print(f'Study years: {STUDY_YEARS}')
print(f'Registry variables: {len(VARIABLE_REGISTRY)}')"""
        break

# Update load_year to use pd.read_excel and better RENAME_MAP
for cell in nb.cells:
    if 'def load_year(year):' in cell.source:
        cell.source = """def load_year(year):
    \"\"\"
    Load one year's CRVS data, rename columns to canonical names,
    and detect which variables are available.
    \"\"\"
    path = DATA_PATHS[year]
    # Use pd.read_excel for .xlsx files
    df   = pd.read_excel(path)

    # ── Column name harmonisation ──────────────────────────────────────────
    # Add all known raw-file column name variants here
    RENAME_MAP = {
        # Birth order variants
        'Birth Order':        'Birth_Order',
        'birth_order':        'Birth_Order',
        'BirthOrder':         'Birth_Order',
        'BIRTH_ORDER':        'Birth_Order',
        'Birth_Order 2.0':    'Birth_Order',
        'BORDER':             'Birth_Order',
        # Maternal age variants
        'Age of Mother':      'Age_of_Mother',
        'age_mother':         'Age_of_Mother',
        'AgeOfMother':        'Age_of_Mother',
        'AGE_OF_MOTHER':      'Age_of_Mother',
        'Mother_Age':         'Age_of_Mother',
        # Marital status
        'Marital Status':     'Marital_Status',
        'marital_status':     'Marital_Status',
        # Race
        'Race of Mother':     'Race_of_Mother',
        'race_mother':        'Race_of_Mother',
        'Race of Father':     'Race_of_Father',
        'race_father':        'Race_of_Father',
        # Gender
        'Sex':                'Gender',
        'sex':                'Gender',
        'Child_Sex':          'Gender',
        'Gender ':            'Gender',
        # Hospital
        'Hospital or Not':    'Hospital_or_Not',
        'hospital_or_not':    'Hospital_or_Not',
        'Place_of_Delivery':  'Hospital_or_Not',
        # Multiple birth
        'Multiple Birth Status': 'Multiple_Birth_Status',
        'multiple_birth':     'Multiple_Birth_Status',
        'Twin':               'Multiple_Birth_Status',
        # Birth weight
        'Birth Weight':       'Birth_Weight_g',
        'birth_weight':       'Birth_Weight_g',
        'BirthWeight':        'Birth_Weight_g',
        'Birth_Weight':       'Birth_Weight_g',
        'BIRTH_WEIGHT':       'Birth_Weight_g',
        'Birth_Weight(grams)': 'Birth_Weight_g',
        # Districts
        'District of Mother': 'District_of_Mother',
        'district_mother':    'District_of_Mother',
        'Registered District':'Registered_District',
        'Registered_District':'Registered_District',
        'reg_district':       'Registered_District',
    }

    df.rename(columns=RENAME_MAP, inplace=True)

    # Strip whitespace from string columns
    str_cols = df.select_dtypes('object').columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # Detect which registry variables are present
    detected = sorted([var for var in VARIABLE_REGISTRY
                       if var in df.columns])
    absent   = sorted([var for var in VARIABLE_REGISTRY
                       if var not in df.columns])

    section_header(f'DATA LOADED: {year}')
    print(f'  Raw records:       {fmt(len(df))}')
    print(f'  Raw columns:       {len(df.columns)}')
    print(f'  Registry detected: {len(detected)} / {len(VARIABLE_REGISTRY)}')
    print(f'  Present:  {detected}')
    print(f'  Absent:   {absent}')

    df.attrs['year']     = year
    df.attrs['detected'] = set(detected)
    return df"""
        break

with open('descriptive_analysis_crvs_2000_2020.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
