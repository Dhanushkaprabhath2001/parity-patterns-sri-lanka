import json

with open('models(2020)_revised.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if 'import dataframe_image' in line:
                new_source.append("# " + line)
            elif 'dfi.export' in line:
                new_source.append("# " + line)
            else:
                new_source.append(line)
        cell['source'] = new_source

with open('models(2020)_revised.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Removed dataframe_image calls from models(2020)_revised.ipynb")
