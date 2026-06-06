import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel

# 1. Load and Preprocess Data
df = pd.read_csv('merged_births_2000_2020.csv')
df = df[(df['Age of Mother'] >= 10) & (df['Age of Mother'] <= 60)].dropna()
order_map = {'First': 1, 'Second': 2, 'Third': 3, 'Fourth': 4, 'Fifth': 5, 'Sixth': 6, 'Seventh': 7, 'Eighth': 8, 'Nineth': 9}
df['Parity_Num'] = df['Birth_Order'].map(order_map)
df = df.dropna(subset=['Parity_Num'])
bins = [0, 20, 25, 30, 35, 100]; labels = ['<20', '20-24', '25-29', '30-34', '35+']
df['Age_Grp'] = pd.cut(df['Age of Mother'], bins=bins, labels=labels, right=False)
main_races = ['Sinhalese', 'Srilankan Tamil', 'Srilankan Moor', 'Indian Tamil']
df['Race_Grp'] = df['Race_of_Mother'].apply(lambda x: x if x in main_races else 'Other')
df['Hospital_or_Not'] = df['Hospital or Not'].replace({'Hospital': 'Hospital', 'Not in Hospital': 'Non-Hospital'})

# Prepare for Ordinal - Manual Dummy Creation
X_ordinal = pd.get_dummies(df[['Age_Grp', 'Race_Grp', 'Gender', 'Hospital_or_Not']], drop_first=True, dtype=float)
# Note: drop_first ensures the reference groups are handled
# Ref: Age 25-29, Race Sinhalese, Gender Female, Hospital delivery

df_sample = df.sample(frac=0.05, random_state=42)
X_sample = pd.get_dummies(df_sample[['Age_Grp', 'Race_Grp', 'Gender', 'Hospital_or_Not']], drop_first=False, dtype=float)
# Re-filter to specific references
cols_to_keep = [c for c in X_sample.columns if not any(ref in c for ref in ['25-29', 'Sinhalese', 'Female', 'Hospital_or_Not_Hospital'])]
X_sample_final = X_sample[cols_to_keep]

print(f"Regression Sample Size: {len(df_sample)}")

# --- Model 1: Ordinal ---
print("\nFitting Model 1: Ordinal...")
mod_ordinal = OrderedModel(df_sample['Parity_Num'], X_sample_final, distr='logit')
res_ordinal = mod_ordinal.fit(method='bfgs', disp=False)
or_ordinal = np.exp(res_ordinal.params)

# --- Model 2: Poisson ---
print("\nFitting Model 2: Poisson...")
# Set Reference groups using C() in formula
formula = "Parity_Num ~ C(Age_Grp, Treatment('25-29')) + C(Race_Grp, Treatment('Sinhalese')) + C(Gender, Treatment('Female')) + C(Hospital_or_Not, Treatment('Hospital'))"
mod_poisson = smf.poisson(formula, data=df_sample).fit(disp=False)
irr_poisson = np.exp(mod_poisson.params)

# --- Model 3: Binary ---
print("\nFitting Model 3: Binary...")
df_sample['Is_First'] = (df_sample['Parity_Num'] == 1).astype(int)
mod_binary = smf.logit("Is_First ~ C(Age_Grp, Treatment('25-29')) + C(Race_Grp, Treatment('Sinhalese')) + C(Gender, Treatment('Female')) + C(Hospital_or_Not, Treatment('Hospital'))", data=df_sample).fit(disp=False)
or_binary = np.exp(mod_binary.params)

# --- Save ---
with open('regression_results.txt', 'w') as f:
    f.write("=== REGRESSION RESULTS ===\n\n")
    f.write("--- Model 1: Ordinal (OR) ---\n" + or_ordinal.to_string() + "\nAIC: " + str(res_ordinal.aic) + "\n\n")
    f.write("--- Model 2: Poisson (IRR) ---\n" + irr_poisson.to_string() + "\nAIC: " + str(mod_poisson.aic) + "\n\n")
    f.write("--- Model 3: Binary (OR) ---\n" + or_binary.to_string() + "\nAIC: " + str(mod_binary.aic) + "\n")

print("\nSuccess. Results in regression_results.txt")
