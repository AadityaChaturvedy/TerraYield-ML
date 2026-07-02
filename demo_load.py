import os
import pandas as pd
import matplotlib.pyplot as plt

# Resolve paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Support both repository layout (data/processed/) and Zenodo package layout (data/)
yield_path_repo = os.path.join(script_dir, "data/processed/crop_rice_season_year_wide_harmonized.csv")
if os.path.exists(yield_path_repo):
    yield_path = yield_path_repo
    climate_path = os.path.join(script_dir, "data/processed/district_climate_data_1997_2022.csv")
else:
    yield_path = os.path.join(script_dir, "data/crop_rice_season_year_wide_harmonized.csv")
    climate_path = os.path.join(script_dir, "data/district_climate_data_1997_2022.csv")

print("Loading datasets from package...")
df_yield = pd.read_csv(yield_path)
df_climate = pd.read_csv(climate_path)

# Merge datasets
print("Merging yield and climate datasets...")
df_yield['State_key'] = df_yield['State'].str.strip().str.lower()
df_yield['District_key'] = df_yield['District'].str.strip().str.lower()

df_climate['State_key'] = df_climate['NAME_1'].str.strip().str.lower()
df_climate['District_key'] = df_climate['NAME_2'].str.strip().str.lower()

df_merged = pd.merge(df_yield, df_climate, on=['State_key', 'District_key'], how='inner')
print(f"Merge successful! Merged shape: {df_merged.shape}")

# Filter to a specific district (Anantapur, Andhra Pradesh)
district_name = "Anantapur"
state_name = "Andhra Pradesh"
df_dist = df_merged[(df_merged['State'] == state_name) & (df_merged['District'] == district_name)].copy()

if df_dist.empty:
    print(f"District {district_name} not found, selecting first available district...")
    state_name = df_merged['State'].iloc[0]
    district_name = df_merged['District'].iloc[0]
    df_dist = df_merged[(df_merged['State'] == state_name) & (df_merged['District'] == district_name)].copy()

df_dist = df_dist.sort_values(by='Year')
print(f"Plotting yield time series for {district_name}, {state_name}...")

plt.figure(figsize=(10, 5))
plt.plot(df_dist['Year'], df_dist['Kharif_Yield'], marker='o', color='teal', linewidth=2)
plt.title(f"Kharif Rice Yield Time Series - {district_name}, {state_name}")
plt.xlabel("Year")
plt.ylabel("Yield (tonnes/ha)")
plt.grid(True, linestyle='--', alpha=0.6)

plot_path = os.path.join(script_dir, "demo_plot.png")
plt.savefig(plot_path, dpi=150)
print(f"Plot saved successfully to {plot_path}!")
