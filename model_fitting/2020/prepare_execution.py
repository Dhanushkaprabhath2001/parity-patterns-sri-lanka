import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import warnings

# Load the revised notebook
with open('models(2020)_revised.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Extract code from all cells
all_code = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if isinstance(source, list):
            source = "".join(source)
        all_code.append(source)

# Join all code
full_script = "\n# " + "="*50 + "\n".join(all_code)

# Write to a python file
with open('run_analysis.py', 'w', encoding='utf-8') as f:
    f.write(full_script)

print("Analysis script 'run_analysis.py' generated.")
