from src.config import CROP_PROFILES

def build_feature_list(
    df,
    crop_profile_name="kharif_rice",
    include_district_feature=False,
    include_extended_months=False,
    include_soil_l1=False,
    include_interactions=False,
    include_yield_trend=False,
    include_lag3=False,
    include_phenology=False,
    include_sensor_flag=False,
    include_yield_lags=True
):
    """
    Build the list of features based on config options.
    Dynamically maps variable names to columns present in the dataframe.
    """
    if crop_profile_name not in CROP_PROFILES:
        raise ValueError(f"Crop profile '{crop_profile_name}' not defined in config.")
    profile = CROP_PROFILES[crop_profile_name]
    
    target_col = profile["target_col"]
    season_prefix = profile["season_prefix"]
    month_labels = profile["month_labels"] if include_extended_months else profile["core_months"]
    core_months = profile["core_months"]

    features = ["State", "Year", "Post2019"]
    if include_yield_lags:
        features.extend([
            f"{target_col}_Lag1", f"{target_col}_Lag2",
            f"{target_col}_Hist_Mean", f"{target_col}_Lag1_Anomaly",
            f"{target_col}_Spatial_Lag1"
        ])
    if include_sensor_flag:
        features.append("Post2000")
    if include_lag3 and include_yield_lags:
        features.append(f"{target_col}_Lag3")
        
    area_col = f"{season_prefix}_Area_Lag1"
    prod_col = f"{season_prefix}_Production_Lag1"
    if area_col in df.columns:
        features.append(area_col)
    if prod_col in df.columns:
        features.append(prod_col)

    if include_district_feature:
        features.append("District_mapped")

    if include_yield_trend and "Yield_Trend" in df.columns:
        features.append("Yield_Trend")

    # Climate raw features
    for m in month_labels:
        features.append(f"Precip_{m}")
        features.append(f"Temp_{m}")
        features.append(f"Soil_{m}")
        if include_soil_l1 and f"SoilL1_{m}" in df.columns:
            features.append(f"SoilL1_{m}")

    # NDVI/EVI raw features
    for m in core_months:
        if f"NDVI_{m}" in df.columns:
            features.append(f"NDVI_{m}")
        if f"EVI_{m}" in df.columns:
            features.append(f"EVI_{m}")

    # Anomaly features
    for var in ["Precip", "Temp", "Soil"]:
        for m in month_labels:
            features.append(f"{var}_{m}_Anomaly")
            
    if include_soil_l1:
        for m in month_labels:
            col = f"SoilL1_{m}_Anomaly"
            if col in df.columns:
                features.append(col)
                
    for var in ["NDVI", "EVI"]:
        for m in core_months:
            col = f"{var}_{m}_Anomaly"
            if col in df.columns:
                features.append(col)

    # Z-Score features
    for var in ["Precip", "Temp", "Soil"]:
        for m in month_labels:
            features.append(f"{var}_{m}_Z")
            
    if include_soil_l1:
        for m in month_labels:
            col = f"SoilL1_{m}_Z"
            if col in df.columns:
                features.append(col)
                
    for var in ["NDVI", "EVI"]:
        for m in core_months:
            col = f"{var}_{m}_Z"
            if col in df.columns:
                features.append(col)

    # CWSI features
    for m in month_labels:
        col = f"CWSI_{m}"
        if col in df.columns:
            features.append(col)

    # Interaction features
    if include_interactions:
        interaction_cols = [
            "Precip_Total_JJAS", "Temp_Mean_JJAS",
            "NDVI_Growth", "EVI_Growth",
            "Drought_Jul", "Precip_Early_Late_Ratio",
            "Soil_Range_JJAS"
        ]
        features += [col for col in interaction_cols if col in df.columns]

    # Include phenology-aligned features if they exist in the dataframe
    if include_phenology:
        phenology_aligned_cols = [
            c for c in df.columns 
            if any(c.endswith(suffix) for suffix in ["_Peak", "_Peak_Minus1", "_Peak_Minus2", "_Peak_Plus1"])
        ]
        features.extend(phenology_aligned_cols)

    return [f for f in features if f in df.columns]
