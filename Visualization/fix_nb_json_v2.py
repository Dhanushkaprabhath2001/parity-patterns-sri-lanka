import json

file_path = 'descriptive_analysis_crvs_2000_2020.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# I'll find the cell with 'Helper functions defined.' and completely rewrite its source
# by finding the 'source': [ block.

start_line = -1
end_line = -1
for i, line in enumerate(lines):
    if 'def normality_check(df, varname):' in line:
        # Trace back to find "source": [
        for j in range(i, 0, -1):
            if '"source": [' in lines[j]:
                start_line = j + 1
                break
        # Trace forward to find ]
        for j in range(i, len(lines)):
            if ']' in lines[j] and 'outputs' in lines[j+1]:
                end_line = j
                break
        break

if start_line != -1 and end_line != -1:
    code_lines = [
        "def has_var(df, varname):\\n",
        "    \\\"\\\"\\\"Return True if varname is a column in df.\\\"\\\"\\\"\\n",
        "    return varname in df.columns\\n",
        "\\n",
        "\\n",
        "def cramers_v(contingency_table):\\n",
        "    \\\"\\\"\\\"Compute Cramér's V from a contingency table (pd.DataFrame or np.array).\\\"\\\"\\\"\\n",
        "    chi2 = chi2_contingency(contingency_table, correction=False)[0]\\n",
        "    n    = np.array(contingency_table).sum()\\n",
        "    r, k = contingency_table.shape\\n",
        "    return np.sqrt(chi2 / (n * (min(r, k) - 1)))\\n",
        "\\n",
        "\\n",
        "def bw_category(bw):\\n",
        "    \\\"\\\"\\\"Assign WHO birth weight category label (units: grams).\\\"\\\"\\\"\\n",
        "    if pd.isna(bw):       return np.nan\\n",
        "    if   bw < BW_ELBW:    return 'ELBW (<1000 g)'\\n",
        "    elif bw < BW_VLBW:    return 'VLBW (1000–1499 g)'\\n",
        "    elif bw < BW_LBW:     return 'LBW (1500–2499 g)'\\n",
        "    elif bw < BW_NORMAL_HI: return 'Normal (2500–3999 g)'\\n",
        "    else:                  return 'Macrosomia (≥4000 g)'\\n",
        "\\n",
        "\\n",
        "def age_group(age):\\n",
        "    \\\"\\\"\\\"Assign WHO-standard maternal age group (completed years).\\\"\\\"\\\"\\n",
        "    if pd.isna(age): return np.nan\\n",
        "    if   age < 20:   return '<20'\\n",
        "    elif age < 25:   return '20–24'\\n",
        "    elif age < 30:   return '25–29'\\n",
        "    elif age < 35:   return '30–34'\\n",
        "    else:            return '35+'\\n",
        "\\n",
        "\\n",
        "def fmt(n):\\n",
        "    \\\"\\\"\\\"Format integer with thousand separators.\\\"\\\"\\\"\\n",
        "    return f'{n:,}'\\n",
        "\\n",
        "\\n",
        "def section_header(title, year=None):\\n",
        "    \\\"\\\"\\\"Print a clear section divider.\\\"\\\"\\\"\\n",
        "    yr = f' — {year}' if year else ''\\n",
        "    print(f\\\"\\\\n{'═'*70}\\\")\\n",
        "    print(f'  {title}{yr}')\\n",
        "    print(f\\\"{'═'*70}\\\")\\n",
        "\\n",
        "\\n",
        "def normality_check(df, varname):\\n",
        "    \\\"\\\"\\\"Perform normality check: Skewness, Kurtosis, and Histogram.\\\"\\\"\\\"\\n",
        "    if not has_var(df, varname):\\n",
        "        return\\n",
        "    \\n",
        "    year = df.attrs.get('year', '?')\\n",
        "    data = df[varname].dropna()\\n",
        "    \\n",
        "    skew = data.skew()\\n",
        "    kurt = data.kurt()\\n",
        "    \\n",
        "    section_header(f'NORMALITY CHECK: {varname}', year)\\n",
        "    print(f'  Skewness: {skew:.3f}')\\n",
        "    print(f'  Kurtosis: {kurt:.3f}')\\n",
        "    \\n",
        "    # Interpretation\\n",
        "    if abs(skew) < 0.5: skew_msg = \\\"Fairly symmetrical\\\"\\n",
        "    elif abs(skew) < 1: skew_msg = \\\"Moderately skewed\\\"\\n",
        "    else: skew_msg = \\\"Highly skewed\\\"\\n",
        "    \\n",
        "    print(f'  Interpretation: {skew_msg}')\\n",
        "    \\n",
        "    plt.figure(figsize=(8, 4))\\n",
        "    sns.histplot(data, kde=True, color='teal')\\n",
        "    plt.title(f'Distribution of {varname} ({year})')\\n",
        "    plt.show()\\n",
        "print('Helper functions defined.')"
    ]
    
    new_source = [f'    "{line}"' for line in code_lines]
    lines[start_line:end_line] = [',\n'.join(new_source) + '\n']
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("Fixed JSON source")
else:
    print("Could not find markers")
