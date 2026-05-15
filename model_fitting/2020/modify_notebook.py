import json
import os
import numpy as np

with open('models(2020).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Add mappings and new columns (after cell 12)
new_data_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 2.5 Revised Mappings & Columns based on Supervisor Feedback\n",
        "\n",
        "# 2.5.1 Collapse Parity: 1, 2, 3, 4+\n",
        "# This addresses the comment: \"do we have to go 1 to 9? can we create 1, 2, 3, and 4<.\"\n",
        "df['parity_collapsed'] = df['parity'].clip(upper=4).astype(int)\n",
        "PARITY_LABELS_COLLAPSED = ['1st', '2nd', '3rd', '4th+']\n",
        "\n",
        "# 2.5.2 Collapse Multiple Birth: Singleton vs Multiple\n",
        "# This addresses the comment: \"if too high have single births and multiple births only; as a binary?\"\n",
        "df['multiple_birth_binary'] = df['multiple_birth'].apply(lambda x: 'Single' if x == 'Singleton' else 'Multiple')\n",
        "\n",
        "# 2.5.3 Province Mapping from District\n",
        "# This addresses the comment: \"can we include the province or a similar indicator for residence?\"\n",
        "DISTRICT_PROVINCE_MAP = {\n",
        "    'Colombo': 'Western', 'Gampaha': 'Western', 'Kalutara': 'Western',\n",
        "    'Kandy': 'Central', 'Matale': 'Central', 'Nuwara Eliya': 'Central',\n",
        "    'Galle': 'Southern', 'Matara': 'Southern', 'Hambantota': 'Southern',\n",
        "    'Jaffna': 'Northern', 'Kilinochchi': 'Northern', 'Mannar': 'Northern', 'Vavuniya': 'Northern', 'Mullaitivu': 'Northern',\n",
        "    'Batticaloa': 'Eastern', 'Ampara': 'Eastern', 'Trincomalee': 'Eastern',\n",
        "    'Kurunegala': 'North Western', 'Puttalam': 'North Western',\n",
        "    'Anuradhapura': 'North Central', 'Polonnaruwa': 'North Central',\n",
        "    'Badulla': 'Uva', 'Moneragala': 'Uva',\n",
        "    'Ratnapura': 'Sabaragamuwa', 'Kegalle': 'Sabaragamuwa'\n",
        "}\n",
        "df['province_mother'] = df['district_mother'].map(DISTRICT_PROVINCE_MAP)\n",
        "\n",
        "print('New columns created:')\n",
        "print(df[['parity_collapsed', 'multiple_birth_binary', 'province_mother']].head())\n",
        "print('\\nCollapsed Parity Distribution:')\n",
        "print(df['parity_collapsed'].value_counts().sort_index())"
    ]
}

nb['cells'].insert(13, new_data_cell)

# Indices shifts by 1
# Original 29 -> 30
# Original 30 -> 31
# Original 31 -> 32
# Original 32 -> 33

# 2. Update Feature Matrix (Cell 30)
nb['cells'][30]['source'] = [
    "# 4.2  Build encoded feature matrix (Revised)\n",
    "FEATURES = ['age_group', 'marital_status', 'race_mother',\n",
    "            'Gender ', 'Hospital or Not', 'multiple_birth_binary', 'province_mother']\n",
    "\n",
    "X_encoded = pd.get_dummies(df[FEATURES], drop_first=True).astype(float)\n",
    "X_encoded['Birth_Weight(grams)'] = df['Birth_Weight(grams)'].values\n",
    "\n",
    "print(f'Feature matrix: {X_encoded.shape[0]:,} rows x {X_encoded.shape[1]} features')\n",
    "print('\\nFeatures:')\n",
    "for f in X_encoded.columns:\n",
    "    print(f'  {f}')"
]

# 3. GVIF Implementation (Cell 31)
gvif_code = [
    "# 4.3  Generalized Variance Inflation Factor (GVIF)\n",
    "# This addresses the comment: \"VIF should be checked only for quantitative variables... use GVIF\"\n",
    "def calculate_gvif(X, predictors):\n",
    "    \"\"\"Calculates GVIF for groups of dummy variables.\"\"\"\n",
    "    import numpy as np\n",
    "    import pandas as pd\n",
    "    \n",
    "    R = X.corr().values\n",
    "    gvif_results = []\n",
    "    \n",
    "    groups = {}\n",
    "    for p in predictors:\n",
    "        groups[p] = [c for c in X.columns if c.startswith(p + '_') or c == p]\n",
    "    \n",
    "    for var, cols in groups.items():\n",
    "        if not cols: continue\n",
    "        indices = [X.columns.get_loc(c) for c in cols]\n",
    "        df_j = len(indices)\n",
    "        \n",
    "        R_j = R[np.ix_(indices, indices)]\n",
    "        others = [i for i in range(X.shape[1]) if i not in indices]\n",
    "        R_others = R[np.ix_(others, others)]\n",
    "        \n",
    "        det_R = np.linalg.det(R)\n",
    "        det_Rj = np.linalg.det(R_j)\n",
    "        det_Rothers = np.linalg.det(R_others)\n",
    "        \n",
    "        gvif = (det_Rj * det_Rothers) / det_R\n",
    "        gvif_adj = gvif**(1/(2*df_j))\n",
    "        \n",
    "        gvif_results.append({\n",
    "            'Predictor': var,\n",
    "            'GVIF': gvif,\n",
    "            'df': df_j,\n",
    "            'GVIF^(1/(2*df))': gvif_adj\n",
    "        })\n",
    "    \n",
    "    return pd.DataFrame(gvif_results)\n",
    "\n",
    "gvif_df = calculate_gvif(X_encoded, FEATURES + ['Birth_Weight(grams)'])\n",
    "print('=' * 55)\n",
    "print('GENERALIZED VARIANCE INFLATION FACTORS (GVIF)')\n",
    "print('GVIF^(1/(2*df)) > 2 (~VIF > 4) is a concern')\n",
    "print('=' * 55)\n",
    "print(gvif_df.sort_values('GVIF^(1/(2*df))', ascending=False).to_string(index=False))"
]
nb['cells'][31]['source'] = gvif_code

# Replace Cell 32 with Markdown
nb['cells'][32] = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "**Supervisor Note:** GVIF is used instead of standard VIF for categorical variables to properly assess multicollinearity across group levels. High GVIF categories should be combined (e.g., province or race) if collinearity persists."
    ]
}

# 4. Update Intent Statement (insert before cell 33)
intent_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 5. Statistical Modeling\n",
        "\n",
        "**Note on Study Intent:** The primary objective of these models is **inference**—identifying the effects of various predictors on parity—rather than purely for prediction. Performance measures (AIC, BIC, Accuracy) are provided to assess model fit, but the focus remains on interpreting the coefficients and odds ratios."
    ]
}
nb['cells'].insert(33, intent_cell)

# 5. Update Outcome for Ordinal Regression (Cell 34)
nb['cells'][34]['source'] = [
    "# 5.1  Prepare collapsed outcome (1, 2, 3, 4+)\n",
    "# Addressing multicollinearity by collapsing low-frequency high-parity categories.\n",
    "y_ord = df['parity_collapsed'].values\n",
    "\n",
    "print('Outcome — Collapsed Parity:')\n",
    "for val, cnt in zip(*np.unique(y_ord, return_counts=True)):\n",
    "    label = PARITY_LABELS_COLLAPSED[val-1]\n",
    "    print(f'  {label:8s} (parity {val}): {cnt:,}')"
]

with open('models(2020)_revised.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
