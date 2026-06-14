import json
import pandas as pd
import numpy as np
import pathlib
from src.config import (
    PROJECT_ROOT,
    DEFAULT_CLIMATE_PATH,
    DEFAULT_NEIGHBORS_PATH,
    CROP_PROFILES
)
from src.district_mappings import district_mapping

def load_and_preprocess(
    crop_profile_name="kharif_rice",
    data_path=None,
    climate_path=None,
    neighbors_path=None,
    include_district_feature=False,
    include_extended_months=False,
    include_soil_l1=False,
    include_interactions=False,
    include_yield_trend=False,
    include_lag3=False,
    include_phenology=False
):
    """
    Load crop yield data, merge with remote sensing & climate data, and compute feature lags/anomalies.
    Fully parameterized to handle different crops, seasons, and feature subsets.

    Parameters
    ----------
    crop_profile_name : str, default="kharif_rice"
        Name of the crop profile defined in CROP_PROFILES.
    data_path : str or pathlib.Path, optional
        Path to the crop yield CSV file.
    climate_path : str or pathlib.Path, optional
        Path to the climate GEE aggregated CSV file.
    neighbors_path : str or pathlib.Path, optional
        Path to the district neighbors adjacency JSON.
    include_district_feature : bool, default=False
        Whether to retain District_mapped as a categorical feature.
    include_extended_months : bool, default=False
        Whether to load extended shoulder months climate data.
    include_soil_l1 : bool, default=False
        Whether to load ERA5 Volumetric Soil Moisture Layer 1.
    include_interactions : bool, default=False
        Whether to generate climate interaction features.
    include_yield_trend : bool, default=False
        Whether to compute district-level linear yield trend slopes.
    include_lag3 : bool, default=False
        Whether to compute 3-year temporal yield lag features.
    include_phenology : bool, default=False
        Whether to perform dynamic phenological month alignment relative to peak NDVI.

    Returns
    -------
    pandas.DataFrame
        Preprocessed dataframe with merged weather, yield statistics, lags, and anomalies.

    Mathematical Formulations
    -------------------------
    1. Expanding Climatic Z-Score:
       For a given district d, year y, and month m, the historical anomaly is standardized as:
       Z_{d, y, m} = (x_{d, y, m} - \mu_{d, <y, m}) / (\sigma_{d, <y, m} + \epsilon)
       where \mu and \sigma are expanding historical averages calculated using only years < y.

    2. Crop Water Stress Index (CWSI):
       CWSI_m = Z_Temp_m - (Z_Precip_m + Z_Soil_m)
    """
    # 2. Get crop profile settings
    if crop_profile_name not in CROP_PROFILES:
        raise ValueError(f"Crop profile '{crop_profile_name}' not defined in config.")
    profile = CROP_PROFILES[crop_profile_name]
    
    # Resolve paths
    if not data_path:
        data_path = PROJECT_ROOT / "data" / "processed" / f"crop_{profile['crop_name']}_season_year_wide_harmonized.csv"
    else:
        data_path = pathlib.Path(data_path)
        
    climate_path = pathlib.Path(climate_path) if climate_path else DEFAULT_CLIMATE_PATH
    neighbors_path = pathlib.Path(neighbors_path) if neighbors_path else DEFAULT_NEIGHBORS_PATH
    
    target_col = profile["target_col"]
    season_prefix = profile["season_prefix"]
    clipping_quantile = profile["clipping_quantile"]

    # 3. Load neighbors dictionary
    with open(neighbors_path, 'r') as f:
        neighbors_dict = json.load(f)

    # 4. Load crop yield data
    df = pd.read_csv(data_path)
    df.replace(-1.0, np.nan, inplace=True)
    
    # Clip target to crop-specific upper limit to prevent extreme outliers
    # (Matches thresholds defined in data_ingestion/clean_crop_data.py)
    yield_thresholds = {
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
        'banana': 120.0,
        'coconut': 100.0
    }
    crop_name = profile["crop_name"]
    max_clip = yield_thresholds.get(crop_name, 15.0)
    df[target_col] = df[target_col].clip(upper=max_clip)

    # 5. Map district names to historical boundaries
    def map_district(row):
        state = str(row["State"]).strip().lower()
        dist = str(row["District"]).strip().lower()
        if (state, dist) in district_mapping:
            return district_mapping[(state, dist)]
        return row["District"]

    df["District_mapped"] = df.apply(map_district, axis=1)
    df = df.sort_values(by=["State", "District_mapped", "Year"]).reset_index(drop=True)

    # 6. Temporal yield lags
    df[f"{target_col}_Lag1"] = df.groupby(["State", "District_mapped"])[target_col].shift(1)
    df[f"{target_col}_Lag2"] = df.groupby(["State", "District_mapped"])[target_col].shift(2)
    if include_lag3:
        df[f"{target_col}_Lag3"] = df.groupby(["State", "District_mapped"])[target_col].shift(3)

    # 7. Expanding historical mean yield per district (Anti-Leakage)
    hist_mean_col = f"{target_col}_Hist_Mean"
    
    def calc_expanding_mean(group):
        group = group.sort_values('Year')
        exp_mean = group[target_col].shift(1).expanding().mean()
        # Fallback: fill the first year (which is NaN due to shift) with its own value
        if len(exp_mean) > 0 and pd.isna(exp_mean.iloc[0]):
            exp_mean.iloc[0] = group[target_col].iloc[0]
        return exp_mean

    df[hist_mean_col] = df.groupby(['State', 'District_mapped'], group_keys=False).apply(calc_expanding_mean)

    # 8. Yield anomaly (deviation from historical mean)
    df[f"{target_col}_Lag1_Anomaly"] = df[f"{target_col}_Lag1"] - df[hist_mean_col]

    # 9. Area and Production Lags
    area_col = f"{season_prefix}_Area"
    prod_col = f"{season_prefix}_Production"
    if area_col in df.columns:
        df[f"{area_col}_Lag1"] = df.groupby(["State", "District_mapped"])[area_col].shift(1)
    if prod_col in df.columns:
        df[f"{prod_col}_Lag1"] = df.groupby(["State", "District_mapped"])[prod_col].shift(1)

    # 10. Spatial Lag Calculation (using neighbors dictionary lookup)
    df['State_clean'] = df['State'].astype(str).str.lower().str.strip()
    df['District_clean'] = df['District_mapped'].astype(str).str.lower().str.strip()
    
    state_avg_dict = df.groupby(['State_clean', 'Year'])[target_col].mean().to_dict()
    yield_dict = df.set_index(['State_clean', 'District_clean', 'Year'])[target_col].to_dict()

    spatial_lags = []
    spatial_col = f"{target_col}_Spatial_Lag1"
    
    for idx, row in df.iterrows():
        state_c, dist_c = row['State_clean'], row['District_clean']
        prev_year = row['Year'] - 1
        key = f"{state_c}|{dist_c}"
        
        neighbors = neighbors_dict.get(key, [])
        ny = []
        for ns in neighbors:
            if '|' in ns:
                n_state, n_dist = ns.split('|')[0], ns.split('|')[1]
            else:
                n_state, n_dist = state_c, ns
            val = yield_dict.get((n_state, n_dist, prev_year))
            if val is not None and not pd.isna(val):
                ny.append(val)
                
        if ny:
            spatial_lags.append(np.mean(ny))
        else:
            fb = state_avg_dict.get((state_c, prev_year))
            if fb is not None and not pd.isna(fb):
                spatial_lags.append(fb)
            else:
                ol = row.get(f"{target_col}_Lag1")
                spatial_lags.append(ol if ol is not None and not pd.isna(ol) else np.nan)

    df[spatial_col] = spatial_lags
    df.drop(columns=['State_clean', 'District_clean'], inplace=True)

    # 11. Yield Trend Feature (expanding district-level linear slope)
    if include_yield_trend:
        trend_values = []
        for (state, dist), group in df.groupby(['State', 'District_mapped']):
            group = group.sort_values('Year')
            slopes = []
            for i in range(len(group)):
                past = group.iloc[:i]
                if len(past) >= 3:
                    x = past['Year'].values.astype(float)
                    y_vals = past[target_col].values.astype(float)
                    mask = ~np.isnan(y_vals)
                    if mask.sum() >= 3:
                        coeffs = np.polyfit(x[mask], y_vals[mask], 1)
                        slopes.append(coeffs[0])
                    else:
                        slopes.append(np.nan)
                else:
                    slopes.append(np.nan)
            trend_df = pd.DataFrame({'idx': group.index, 'Yield_Trend': slopes})
            trend_values.append(trend_df)
        trend_all = pd.concat(trend_values)
        df.loc[trend_all['idx'].values, 'Yield_Trend'] = trend_all['Yield_Trend'].values

    # 12. Drop rows missing target or lags
    cols_to_check = [target_col, f"{target_col}_Lag1", f"{target_col}_Lag2", spatial_col]
    if include_lag3:
        cols_to_check.append(f"{target_col}_Lag3")
    if area_col in df.columns:
        cols_to_check.append(f"{area_col}_Lag1")
    if prod_col in df.columns:
        cols_to_check.append(f"{prod_col}_Lag1")
        
    df = df.dropna(subset=cols_to_check).copy()
    df["Post2019"] = (df["Year"] >= 2019).astype(int)
    df["Post2000"] = (df["Year"] >= 2000).astype(int)

    # 13. Load and merge climate data
    if climate_path.exists():
        df_climate = pd.read_csv(climate_path)
        df_climate.replace(-9999.0, np.nan, inplace=True)
        climate_records = []

        # Determine which months to extract
        if season_prefix.lower() == "kharif":
            month_offsets = profile["month_offsets"] if include_extended_months else [5, 6, 7, 8]
            month_labels = profile["month_labels"] if include_extended_months else profile.get("core_months", [])
        else:
            # For Rabi or other seasons, use the defined offsets directly
            month_offsets = profile["month_offsets"]
            month_labels = profile["month_labels"]

        for _, row in df_climate.iterrows():
            state = row.get("NAME_1")
            district = row.get("NAME_2")
            if pd.isna(state) or pd.isna(district):
                continue

            for year in range(1997, 2023):
                offset = (year - 1997) * 12
                try:
                    rec = {"State": state, "District_mapped": district, "Year": year}
                    for i, m_off in enumerate(month_offsets):
                        m = offset + m_off
                        name = month_labels[i]
                        rec[f"Precip_{name}"] = row[f"{m}_Precip_Sum"]
                        rec[f"Temp_{name}"] = row[f"{m}_Temp_Mean"]
                        rec[f"Soil_{name}"] = row[f"{m}_Soil_L2"]
                        if include_soil_l1:
                            rec[f"SoilL1_{name}"] = row[f"{m}_Soil_L1"]
                        # NDVI/EVI only available for core Kharif months (Jun-Sep)
                        if season_prefix.lower() == "kharif" and m_off in [5, 6, 7, 8]:
                            rec[f"NDVI_{name}"] = row.get(f"{m}_NDVI", np.nan)
                            rec[f"EVI_{name}"] = row.get(f"{m}_EVI", np.nan)
                    climate_records.append(rec)
                except KeyError:
                    pass

        df_climate_long = pd.DataFrame(climate_records)
        df = pd.merge(df, df_climate_long, on=["State", "District_mapped", "Year"], how="left")

        # Drop rows missing core climate/remote sensing data dynamically
        check_cols = []
        for m in month_labels:
            check_cols.extend([f"Precip_{m}", f"Temp_{m}", f"Soil_{m}"])
        if season_prefix.lower() == "kharif":
            for m in profile.get("core_months", []):
                check_cols.append(f"NDVI_{m}")
                check_cols.append(f"EVI_{m}")
        check_cols = [c for c in check_cols if c in df.columns]
        df.dropna(subset=check_cols, inplace=True)

        # 14. Calculate expanding climate anomalies and standardized z-scores
        all_monthly_cols = []
        for var in ["Precip", "Temp", "Soil"]:
            for m in month_labels:
                all_monthly_cols.append(f"{var}_{m}")
        if include_soil_l1:
            for m in month_labels:
                all_monthly_cols.append(f"SoilL1_{m}")
        for var in ["NDVI", "EVI"]:
            for m in profile.get("core_months", []):
                all_monthly_cols.append(f"{var}_{m}")

        all_monthly_cols = [c for c in all_monthly_cols if c in df.columns]
        new_cols_dict = {}
        
        def calc_expanding_mean(g, col):
            g = g.sort_values('Year')
            res = g[col].shift(1).expanding().mean()
            if len(res) > 0 and pd.isna(res.iloc[0]):
                res.iloc[0] = g[col].iloc[0]
            return res

        def calc_expanding_std(g, col):
            g = g.sort_values('Year')
            res = g[col].shift(1).expanding().std()
            if len(res) > 0 and pd.isna(res.iloc[0]):
                res.iloc[0] = 1.0
            return res.fillna(1.0).replace(0.0, 1.0)

        for col in all_monthly_cols:
            hist_mean_col_name = f'{col}_Hist_Mean'
            hist_std_col_name = f'{col}_Hist_Std'
            anomaly_col_name = f'{col}_Anomaly'
            z_score_col_name = f'{col}_Z'
            
            new_cols_dict[hist_mean_col_name] = df.groupby(['State', 'District_mapped'], group_keys=False).apply(calc_expanding_mean, col=col)
            new_cols_dict[hist_std_col_name] = df.groupby(['State', 'District_mapped'], group_keys=False).apply(calc_expanding_std, col=col)
            
            new_cols_dict[anomaly_col_name] = df[col] - new_cols_dict[hist_mean_col_name]
            new_cols_dict[z_score_col_name] = new_cols_dict[anomaly_col_name] / (new_cols_dict[hist_std_col_name] + 1e-5)

        # 14b. Combined Crop Water Stress Index (CWSI) proxy per month
        for m in month_labels:
            t_col = f"Temp_{m}_Z"
            p_col = f"Precip_{m}_Z"
            s_col = f"Soil_{m}_Z"
            
            # Since these are newly added, they are in new_cols_dict
            if t_col in new_cols_dict and p_col in new_cols_dict and s_col in new_cols_dict:
                new_cols_dict[f"CWSI_{m}"] = new_cols_dict[t_col] - (new_cols_dict[p_col] + new_cols_dict[s_col])
                
        # Concatenate all new columns at once to prevent Pandas fragmentation warnings
        if new_cols_dict:
            df = pd.concat([df, pd.DataFrame(new_cols_dict)], axis=1)

        # 14c. Dynamic Phenology Alignment (for Kharif crops with NDVI)
        ndvi_cols = [f"NDVI_{m}" for m in ["Jun", "Jul", "Aug", "Sep"]]
        if include_phenology and all(col in df.columns for col in ndvi_cols):
            df["Peak_Month"] = df[ndvi_cols].idxmax(axis=1).str.replace("NDVI_", "")
            
            alignment_map = {
                "Jun": {"Peak": "Jun", "Peak_Minus1": "May", "Peak_Minus2": "May", "Peak_Plus1": "Jul"},
                "Jul": {"Peak": "Jul", "Peak_Minus1": "Jun", "Peak_Minus2": "May", "Peak_Plus1": "Aug"},
                "Aug": {"Peak": "Aug", "Peak_Minus1": "Jul", "Peak_Minus2": "Jun", "Peak_Plus1": "Sep"},
                "Sep": {"Peak": "Sep", "Peak_Minus1": "Aug", "Peak_Minus2": "Jul", "Peak_Plus1": "Oct"}
            }
            
            vars_to_align = ["Precip", "Temp", "Soil"]
            if include_soil_l1:
                vars_to_align.append("SoilL1")
                
            for base_var in vars_to_align:
                for suffix in ["", "_Anomaly", "_Z"]:
                    var = f"{base_var}{suffix}"
                    for pos in ["Peak", "Peak_Minus1", "Peak_Minus2", "Peak_Plus1"]:
                        conditions = []
                        choices = []
                        for peak_m, mapping in alignment_map.items():
                            actual_m = mapping[pos]
                            actual_col = f"{base_var}_{actual_m}{suffix}"
                            if actual_col in df.columns:
                                conditions.append(df["Peak_Month"] == peak_m)
                                choices.append(df[actual_col])
                        if conditions:
                            df[f"{var}_{pos}"] = np.select(conditions, choices, default=np.nan)
            
            # Align CWSI
            for pos in ["Peak", "Peak_Minus1", "Peak_Minus2", "Peak_Plus1"]:
                conditions = []
                choices = []
                for peak_m, mapping in alignment_map.items():
                    actual_m = mapping[pos]
                    actual_col = f"CWSI_{actual_m}"
                    if actual_col in df.columns:
                        conditions.append(df["Peak_Month"] == peak_m)
                        choices.append(df[actual_col])
                if conditions:
                    df[f"CWSI_{pos}"] = np.select(conditions, choices, default=np.nan)
                    
            # Align NDVI and EVI
            alignment_map_veg = {
                "Jun": {"Peak": "Jun", "Peak_Minus1": "Jun", "Peak_Plus1": "Jul"},
                "Jul": {"Peak": "Jul", "Peak_Minus1": "Jun", "Peak_Plus1": "Aug"},
                "Aug": {"Peak": "Aug", "Peak_Minus1": "Jul", "Peak_Plus1": "Sep"},
                "Sep": {"Peak": "Sep", "Peak_Minus1": "Aug", "Peak_Plus1": "Sep"}
            }
            for base_var in ["NDVI", "EVI"]:
                for suffix in ["", "_Anomaly", "_Z"]:
                    var = f"{base_var}{suffix}"
                    for pos in ["Peak", "Peak_Minus1", "Peak_Plus1"]:
                        conditions = []
                        choices = []
                        for peak_m, mapping in alignment_map_veg.items():
                            actual_m = mapping[pos]
                            actual_col = f"{base_var}_{actual_m}{suffix}"
                            if actual_col in df.columns:
                                conditions.append(df["Peak_Month"] == peak_m)
                                choices.append(df[actual_col])
                        if conditions:
                            df[f"{var}_{pos}"] = np.select(conditions, choices, default=np.nan)

        # 15. Interaction features
        if include_interactions:
            df["Precip_Total_JJAS"] = df[[f"Precip_{m}" for m in ["Jun", "Jul", "Aug", "Sep"]]].sum(axis=1)
            df["Temp_Mean_JJAS"] = df[[f"Temp_{m}" for m in ["Jun", "Jul", "Aug", "Sep"]]].mean(axis=1)
            df["NDVI_Growth"] = df["NDVI_Sep"] - df["NDVI_Jun"]
            df["EVI_Growth"] = df["EVI_Sep"] - df["EVI_Jun"]
            df["Drought_Jul"] = df["Precip_Jul"] / (df["Temp_Jul"] + 0.001)
            df["Precip_Early_Late_Ratio"] = df["Precip_Jun"] / (df["Precip_Sep"] + 0.001)
            df["Soil_Range_JJAS"] = df[[f"Soil_{m}" for m in ["Jun", "Jul", "Aug", "Sep"]]].max(axis=1) - \
                                    df[[f"Soil_{m}" for m in ["Jun", "Jul", "Aug", "Sep"]]].min(axis=1)

    df["State"] = df["State"].astype("category")
    df["District"] = df["District"].astype("category")
    df["District_mapped"] = df["District_mapped"].astype("category")

    return df
