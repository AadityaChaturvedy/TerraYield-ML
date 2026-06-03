"""
Clean and Harmonize Crop Yield Data.
Addresses unit inconsistencies (coconuts), extreme yield outliers (data entry errors),
and drops columns containing only missing value markers (-1.0).
"""
import pandas as pd
import numpy as np
import glob
import pathlib

# Thresholds for maximum realistic yield (tons/ha or equivalent) per crop
YIELD_THRESHOLDS = {
    'rice': 10.0,
    'wheat': 10.0,
    'maize': 15.0,
    'onion': 100.0,
    'potato': 100.0,
    'sugarcane': 250.0,
    'arhar_tur': 8.0,
    'groundnut': 10.0,
    'soyabean': 8.0,
    'cotton_lint': 15.0,
    'ginger': 50.0,
    'turmeric': 30.0,
    'tobacco': 10.0,
    'banana': 120.0
}

def clean_file(file_path):
    print(f"Cleaning: {file_path.name}")
    df = pd.read_csv(file_path)
    
    crop_name = file_path.name.split('_')[1]
    
    # 1. Address Coconut Unit Mismatch (Convert from single nuts to thousands of nuts)
    if crop_name == "coconut":
        print("  -> Normalizing Coconut production and yield by /1000 (thousands of nuts)")
        for col in df.columns:
            if any(s in col for s in ["Production", "Yield"]) and df[col].dtype in [np.float64, np.int64]:
                # Only scale valid positive values
                mask = df[col] > 0
                df.loc[mask, col] = df.loc[mask, col] / 1000.0
                
    # 2. Address Extreme Yield Outliers
    if crop_name in YIELD_THRESHOLDS:
        thresh = YIELD_THRESHOLDS[crop_name]
        yield_cols = [c for c in df.columns if 'Yield' in c]
        for y_col in yield_cols:
            mask = df[y_col] > thresh
            if mask.any():
                count = mask.sum()
                max_val = df.loc[mask, y_col].max()
                print(f"  -> Found {count} outliers in {y_col} (max: {max_val:.2f} > threshold: {thresh})")
                
                # Invalidate yield and production for these outliers (set to -1.0)
                df.loc[mask, y_col] = -1.0
                p_col = y_col.replace("Yield", "Production")
                if p_col in df.columns:
                    df.loc[mask, p_col] = -1.0
                    
    # 3. Identify and drop empty/missing-only columns (max == -1.0)
    cols_to_drop = []
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            if df[col].max() == -1.0:
                cols_to_drop.append(col)
                
    if cols_to_drop:
        print(f"  -> Dropping empty seasonal columns: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)
        
    # Round all yield and production columns for formatting
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].round(4)
    
    # Save back to CSV
    df.to_csv(file_path, index=False)

def main():
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "processed"
    
    # Find all crop yield files
    crop_files = list(data_dir.glob("crop_*_season_year_wide_harmonized.csv"))
    print(f"Found {len(crop_files)} harmonized files to clean in {data_dir}")
    
    for f in crop_files:
        clean_file(f)
        
    print("\nData cleaning and unit normalization complete!")

if __name__ == "__main__":
    main()
