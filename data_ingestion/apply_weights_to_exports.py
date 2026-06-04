import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # 1. Define paths
    dataset_dir = Path(__file__).resolve().parent.parent / "data"
    exports_dir = dataset_dir / "raw"
    out_dir = dataset_dir / "processed"
    weights_path = dataset_dir / "processed" / "clean_india_district_conversion_weights.csv"
    
    # Create output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading conversion weights...")
    df_weights = pd.read_csv(weights_path)
    
    df_weights["merge_state"] = df_weights["state_name"].str.lower().str.strip()
    df_weights["merge_district"] = df_weights["old_district"].str.lower().str.strip()
    
    # 2. Find all crop CSVs
    crop_files = list(exports_dir.glob("crop_*_season_year_wide.csv"))
    print(f"Found {len(crop_files)} raw crop files to process.")
    
    # Define columns
    seasons = ["Rabi", "Kharif", "Autum", "Winter", "Summer", "Year"]
    extensive_cols = []
    for s in seasons:
        extensive_cols.append(f"{s}_Area")
        extensive_cols.append(f"{s}_Production")
        
    # Dictionary of district name corrections for truncated names
    corrections = {
        "Sri Potti Sriramulu Nell*": "Sri Potti Sriramulu Nellore",
        "Sahibzada Ajit Singh Nag*": "Sahibzada Ajit Singh Nagar",
        "North Twenty Four Pargan*": "North Twenty Four Parganas"
    }
        
    for f in crop_files:
        print(f"Processing: {f.name}")
        df_raw = pd.read_csv(f)
        
        # A. Handle -1.0 missing values -> Convert to NaN
        num_cols = df_raw.select_dtypes(include=[np.number]).columns
        df_raw[num_cols] = df_raw[num_cols].replace(-1.0, np.nan)
        
        # B. Prepare for merge
        df_raw["merge_state"] = df_raw["State"].str.lower().str.strip()
        df_raw["merge_district"] = df_raw["District"].str.lower().str.strip()
        
        # Merge weights
        df_merged = pd.merge(df_raw, df_weights, on=["merge_state", "merge_district"], how="inner")
        
        # C. Apply weights to extensive variables
        for col in extensive_cols:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col] * df_merged["weight_old_to_new"]
                
        # D. Re-aggregate back to the NEW district boundaries
        group_cols = ["Crop_Name", "state_name", "new_district", "Year"]
        
        agg_dict = {col: lambda x: x.sum(min_count=1) for col in extensive_cols if col in df_merged.columns}
        
        df_agg = df_merged.groupby(group_cols).agg(agg_dict).reset_index()
        
        # Rename columns to match original
        df_agg = df_agg.rename(columns={
            "state_name": "State",
            "new_district": "District"
        })
        
        # E. Fix Truncated Names
        df_agg["District"] = df_agg["District"].replace(corrections)
        
        # F. Recalculate Intensive Variables (Yield)
        for s in seasons:
            area_col = f"{s}_Area"
            prod_col = f"{s}_Production"
            yield_col = f"{s}_Yield"
            
            if area_col in df_agg.columns and prod_col in df_agg.columns:
                df_agg[yield_col] = np.where(df_agg[area_col] > 0, df_agg[prod_col] / df_agg[area_col], np.nan)
                
        # G. Reorder columns to match original structure
        original_cols = [c for c in df_raw.columns if c not in ["merge_state", "merge_district"]]
        final_cols = [c for c in original_cols if c in df_agg.columns]
        df_agg = df_agg[final_cols]
        
        # Round numeric columns for neatness
        df_agg[final_cols[4:]] = df_agg[final_cols[4:]].round(4)
        
        # H. Restore missing values (-1.0)
        df_agg = df_agg.fillna(-1.0)
        
        # Save to harmonized exports folder
        out_path = out_dir / f.name.replace(".csv", "_harmonized.csv")
        df_agg.to_csv(out_path, index=False)
        
    print("\nAll files successfully harmonized with clean names!")

if __name__ == "__main__":
    main()
