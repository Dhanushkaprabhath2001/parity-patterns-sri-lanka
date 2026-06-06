import os

file_path = 'descriptive_analysis_crvs_2000_2020.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add a normality check function to the Helper Functions cell
normality_func = """def normality_check(df, varname):\\n    \"\"\"Perform normality check: Skewness, Kurtosis, and Histogram.\"\"\"\\n    if not has_var(df, varname):\\n        return\\n    \\n    year = df.attrs.get('year', '?')\\n    data = df[varname].dropna()\\n    \\n    skew = data.skew()\\n    kurt = data.kurt()\\n    \\n    section_header(f'NORMALITY CHECK: {varname}', year)\\n    print(f'  Skewness: {skew:.3f}')\\n    print(f'  Kurtosis: {kurt:.3f}')\\n    \\n    # Interpretation\\n    if abs(skew) < 0.5: skew_msg = \"Fairly symmetrical\"\\n    elif abs(skew) < 1: skew_msg = \"Moderately skewed\"\\n    else: skew_msg = \"Highly skewed\"\\n    \\n    print(f'  Interpretation: {skew_msg}')\\n    \\n    plt.figure(figsize=(8, 4))\\n    sns.histplot(data, kde=True, color='teal')\\n    plt.title(f'Distribution of {varname} ({year})')\\n    plt.show()\\n"""

# Insert it before 'print('Helper functions defined.')'
content = content.replace("print('Helper functions defined.')", normality_func + "print('Helper functions defined.')")

# Call normality_check in the loop
# Find 'geographic_distribution(df)' and add 'normality_check' after it
content = content.replace("geographic_distribution(df)", "geographic_distribution(df)\\n    normality_check(df, 'Age_of_Mother')\\n    normality_check(df, 'Birth_Weight_g')")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated successfully")
