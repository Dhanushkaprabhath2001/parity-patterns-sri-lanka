import os

file_path = 'descriptive_analysis_crvs_2000_2020.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Update DATA_PATHS
old_paths = """    2000: 'data/crvs_2000.csv',
    2005: 'data/crvs_2005.csv',
    2010: 'data/crvs_2010.csv',
    2015: 'data/crvs_2015.csv',
    2020: 'data/crvs_2020.csv',"""
new_paths = """    2000: 'complete_birth_2000.xlsx',
    2005: 'complete_birth_2005.xlsx',
    2010: 'complete_birth_2010.xlsx',
    2015: 'complete_birth_2015.xlsx',
    2020: 'complete_birth_2020.xlsx',"""
content = content.replace(old_paths.replace('\n', '\\n'), new_paths.replace('\n', '\\n'))

# Update pd.read_csv to pd.read_excel
content = content.replace('pd.read_csv(path, low_memory=False)', 'pd.read_excel(path)')

# Update VARIABLE_REGISTRY - this one is trickier due to size, I'll do it carefully
# I'll just replace the whole dictionary content
import re
pattern = r'"VARIABLE_REGISTRY = \{.*?\}"' # This might not work well with multi-line
# Let's use a simpler approach

# Since I know the content from read_file, I'll just use exact string from there but with \n escaped
old_reg = """    'Birth_Order':           [2000, 2005, 2010, 2015, 2020],\\n    'Age_of_Mother':         [2000, 2005, 2010, 2015, 2020],\\n    'Marital_Status':        [2000, 2005, 2010, 2015, 2020],\\n    'Race_of_Mother':        [2000, 2005, 2010, 2015, 2020],\\n    'Gender':                [2000, 2005, 2010, 2015, 2020],\\n    'Hospital_or_Not':       [2000, 2005, 2010, 2015, 2020],\\n    'Multiple_Birth_Status': [2000, 2005, 2010, 2015, 2020],\\n    'Birth_Weight_g':        [2000, 2005, 2010, 2015, 2020],\\n    'District_of_Mother':    [2000, 2005, 2010, 2015, 2020],\\n    'Registered_District':   [2000, 2005, 2010, 2015, 2020],\\n\\n    # Extended variables \\u2014 only in 2010+\\n    'Race_of_Father':        [2010, 2015, 2020],\\n    'Gestational_Age_wk':    [2010, 2015, 2020],\\n\\n    # Further extended \\u2014 only in 2015+\\n    'Maternal_Education':    [2015, 2020],\\n    'Antenatal_Visits':      [2015, 2020],"""
new_reg = """    'Birth_Order':           [2000, 2005, 2010, 2015, 2020],\\n    'Age_of_Mother':         [2000, 2005, 2010, 2015, 2020],\\n    'Marital_Status':        [2000, 2005, 2010, 2015, 2020],\\n    'Race_of_Mother':        [2000, 2005, 2010, 2015, 2020],\\n    'Gender':                [2000, 2005, 2010, 2015, 2020],\\n    'Hospital_or_Not':       [2000, 2005, 2010, 2015, 2020],\\n    'District_of_Mother':    [2000, 2005, 2010, 2015, 2020],\\n    'Registered_District':   [2000, 2005, 2010, 2015, 2020],\\n\\n    # Extended variables \\u2014 only in 2010+\\n    'Multiple_Birth_Status': [2010, 2015, 2020],\\n    'Birth_Weight_g':        [2010, 2015, 2020],\\n    'Race_of_Father':        [2010, 2015, 2020],"""

content = content.replace(old_reg, new_reg)

# Update RENAME_MAP
# I'll just add the new fields to the RENAME_MAP
content = content.replace("'BIRTH_ORDER':        'Birth_Order',", "'BIRTH_ORDER':        'Birth_Order',\\n        'Birth_Order 2.0':    'Birth_Order',\\n        'BORDER':             'Birth_Order',")
content = content.replace("'Child_Sex':          'Gender',", "'Child_Sex':          'Gender',\\n        'Gender ':            'Gender',")
content = content.replace("'multiple_birth':     'Multiple_Birth_Status',", "'multiple_birth':     'Multiple_Birth_Status',\\n        'Twin':               'Multiple_Birth_Status',")
content = content.replace("'BIRTH_WEIGHT':       'Birth_Weight_g',", "'BIRTH_WEIGHT':       'Birth_Weight_g',\\n        'Birth_Weight(grams)': 'Birth_Weight_g',")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated successfully")
