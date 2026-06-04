import pandas as pd
import numpy as np

def main():
    csv_path = 'data/processed/district_climate_data_1997_2022.csv'
    print(f"Loading climate data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print("Imputing pre-2000 MODIS NDVI/EVI columns with district-level climatology (2000-2022 average)...")
    
    # We replace any -9999.0 with NaN first to make mean calculation easy
    df.replace(-9999.0, np.nan, inplace=True)
    
    # Kharif months indices relative to start of year: 5 (June), 6 (July), 7 (August), 8 (September)
    kharif_month_offsets = [5, 6, 7, 8]
    
    # We will do this for both NDVI and EVI
    for feature in ['NDVI', 'EVI']:
        for offset in kharif_month_offsets:
            # 1. Identify all historical columns for this specific month across years 2000 to 2022
            hist_cols = []
            for yr in range(2000, 2023):
                m_idx = (yr - 1997) * 12 + offset
                col_name = f"{m_idx}_{feature}"
                if col_name in df.columns:
                    hist_cols.append(col_name)
            
            # 2. Compute the district-level average for this month
            # (row-wise mean across the years 2000-2022)
            district_means = df[hist_cols].mean(axis=1)
            
            # 3. Impute the pre-2000 years (1997, 1998, 1999)
            for yr in [1997, 1998, 1999]:
                m_idx = (yr - 1997) * 12 + offset
                col_name = f"{m_idx}_{feature}"
                
                # If the column exists and has missing values (NaNs), fill them with the district mean
                if col_name in df.columns:
                    df[col_name] = df[col_name].fillna(district_means)
                    
    # Verify that there are no NaNs in the Kharif NDVI/EVI columns now
    all_kharif_cols = []
    for yr in range(1997, 2023):
        for offset in kharif_month_offsets:
            m_idx = (yr - 1997) * 12 + offset
            all_kharif_cols.extend([f"{m_idx}_NDVI", f"{m_idx}_EVI"])
            
    missing_after = df[all_kharif_cols].isna().sum().sum()
    print(f"Imputation completed. Total missing values in all Kharif MODIS columns: {missing_after}")

    # Save the imputed climate database back to CSV
    print(f"Saving imputed data back to {csv_path}...")
    df.to_csv(csv_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
