import json

file_path = 'descriptive_analysis_crvs_2000_2020.ipynb'

# Since the file is currently invalid JSON, I can't use json.load
# I have to fix it as a string first or revert to a known good state.

# Actually, I'll just write the WHOLE notebook content from scratch if I have to, 
# but it's too big.

# I'll try to fix the problematic part. 
# The issue is I inserted raw Python code into the JSON source list.

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# I'll try to find the broken cell and replace it with properly formatted JSON strings.
# The broken part starts with 'def normality_check' and ends before 'print(\'Helper functions defined.\')'

import re

# This regex tries to find the broken insertion
# It looks for the text I inserted which is not in the "line", "line", format.
pattern = r'def normality_check\(df, varname\):.*?print\(\'Helper functions defined\.\'\)'
# But it's multi-line.

# Let's just use a simpler way: find the cell that contains 'Helper functions defined.'
# and rewrite its 'source' list.

# I'll read the file, and since it's just one cell that's broken, I'll use a regex to find the cell.
# A cell looks like { "cell_type": "code", ..., "source": [ ... ] }

# Actually, I'll just use a python script that treats the file as a notebook and fixes the source.
# I'll use json.loads on a REPAIRED string.

# The repair:
# 1. Replace the raw newlines I introduced with \n
# 2. Ensure it's inside quotes.

# Wait, the easiest way is to use the fact that I know what I was trying to do.
# I'll write a script that reads the file, finds the index of 'def normality_check', 
# and replaces that whole block with a properly escaped JSON string list.

def fix():
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Identify the start and end of the broken insertion
    start_marker = '"def normality_check(df, varname):'
    end_marker = "print('Helper functions defined.')\""
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker) + len(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        print("Markers not found")
        return

    # The code I want to insert, properly escaped for a JSON list of strings
    code_lines = [
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
    
    new_fragment = '    ' + ',\\n    '.join(['"' + line + '"' for line in code_lines])
    
    new_content = content[:start_idx] + new_fragment + content[end_idx:]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Fixed JSON")

fix()
