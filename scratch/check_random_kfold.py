import pandas as pd
import numpy as np
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import pathlib
import json

np.random.seed(42)

# Dynamic paths resolved relative to project root
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "crop_rice_season_year_wide_harmonized.csv"
CLIMATE_PATH = PROJECT_ROOT / "data" / "processed" / "district_climate_data_1997_2022.csv"
NEIGHBORS_PATH = PROJECT_ROOT / "data" / "processed" / "district_neighbors.json"

TARGET_COL = "Kharif_Yield"
SEASON_PREFIX = "Kharif"

def load_and_preprocess() -> pd.DataFrame:
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df.replace(-1.0, np.nan, inplace=True)
    df[TARGET_COL] = df[TARGET_COL].clip(upper=15.0) 
    df = df.sort_values(by=["State", "District", "Year"]).reset_index(drop=True)
    
    df[f"{TARGET_COL}_Lag1"] = df.groupby(["State", "District"])[TARGET_COL].shift(1)
    df[f"{TARGET_COL}_Lag2"] = df.groupby(["State", "District"])[TARGET_COL].shift(2)
    
    area_col_name = f"{SEASON_PREFIX}_Area"
    prod_col_name = f"{SEASON_PREFIX}_Production"
    area_col = area_col_name if area_col_name in df.columns else None
    prod_col = prod_col_name if prod_col_name in df.columns else None
    
    if area_col:
        df[f"{area_col}_Lag1"] = df.groupby(["State", "District"])[area_col].shift(1)
    if prod_col:
        df[f"{prod_col}_Lag1"] = df.groupby(["State", "District"])[prod_col].shift(1)
        
    print("Loading cached neighbors JSON...")
    with open(NEIGHBORS_PATH, 'r') as f:
        neighbors_dict = json.load(f)
        
    df['State_clean'] = df['State'].astype(str).str.lower().str.strip()
    df['District_clean'] = df['District'].astype(str).str.lower().str.strip()
    
    yield_dict = df.set_index(['State_clean', 'District_clean', 'Year'])[TARGET_COL].to_dict()
    state_avg_dict = df.groupby(['State_clean', 'Year'])[TARGET_COL].mean().to_dict()
    
    spatial_lags = []
    for idx, row in df.iterrows():
        state_c = row['State_clean']
        dist_c = row['District_clean']
        year = row['Year']
        prev_year = year - 1
        
        key = f"{state_c}|{dist_c}"
        neighbors = neighbors_dict.get(key, [])
        neighbor_yields = []
        for n_str in neighbors:
            n_state, n_dist = n_str.split('|')
            val = yield_dict.get((n_state, n_dist, prev_year))
            if val is not None and not pd.isna(val):
                neighbor_yields.append(val)
                
        if neighbor_yields:
            spatial_lags.append(np.mean(neighbor_yields))
        else:
            fallback_val = state_avg_dict.get((state_c, prev_year))
            if fallback_val is not None and not pd.isna(fallback_val):
                spatial_lags.append(fallback_val)
            else:
                own_lag = row.get(f"{TARGET_COL}_Lag1")
                if own_lag is not None and not pd.isna(own_lag):
                    spatial_lags.append(own_lag)
                else:
                    spatial_lags.append(np.nan)
                    
    df['Kharif_Yield_Spatial_Lag1'] = spatial_lags
    df.drop(columns=['State_clean', 'District_clean'], inplace=True)

    cols_to_check = [TARGET_COL, f"{TARGET_COL}_Lag1", f"{TARGET_COL}_Lag2", "Kharif_Yield_Spatial_Lag1"]
    if area_col: cols_to_check.append(f"{area_col}_Lag1")
    if prod_col: cols_to_check.append(f"{prod_col}_Lag1")
        
    df = df.dropna(subset=cols_to_check).copy()
    df["State"] = df["State"].astype("category")
    df["District"] = df["District"].astype("category")
    df["Post2019"] = (df["Year"] >= 2019).astype(int)
    
    if CLIMATE_PATH.exists():
        df_climate = pd.read_csv(CLIMATE_PATH)
        df_climate.replace(-9999.0, np.nan, inplace=True)
        climate_records = []
        for idx, row in df_climate.iterrows():
            state = row.get("NAME_1")
            district = row.get("NAME_2")
            if pd.isna(state) or pd.isna(district): continue
            
            for year in range(1997, 2023):
                offset = (year - 1997) * 12
                months = [offset + 5, offset + 6, offset + 7, offset + 8]
                precip_cols = [f"{m}_Precip_Sum" for m in months]
                temp_cols = [f"{m}_Temp_Mean" for m in months]
                soil_l2_cols = [f"{m}_Soil_L2" for m in months]
                try:
                    precip_total = sum(row[c] for c in precip_cols)
                    temp_mean = np.mean([row[c] for c in temp_cols])
                    soil_l2_mean = np.mean([row[c] for c in soil_l2_cols])
                    climate_records.append({
                        "State": state,
                        "District": district,
                        "Year": year,
                        "Kharif_Precip_Total": precip_total,
                        "Kharif_Temp_Mean": temp_mean,
                        "Kharif_Soil_Moist_L2": soil_l2_mean
                    })
                except KeyError:
                    pass
                    
        df_climate_long = pd.DataFrame(climate_records)
        df = pd.merge(df, df_climate_long, on=["State", "District", "Year"], how="left")
        df.dropna(subset=["Kharif_Precip_Total", "Kharif_Temp_Mean", "Kharif_Soil_Moist_L2"], inplace=True)
        df["State"] = df["State"].astype("category")
        df["District"] = df["District"].astype("category")
        
    return df

def main():
    df = load_and_preprocess()
    
    features = [
        "State", "Year", "Post2019",
        f"{TARGET_COL}_Lag1", f"{TARGET_COL}_Lag2",
        "Kharif_Yield_Spatial_Lag1",
        f"{SEASON_PREFIX}_Area_Lag1", f"{SEASON_PREFIX}_Production_Lag1",
        "Kharif_Precip_Total", "Kharif_Temp_Mean",
        "Kharif_Soil_Moist_L2"
    ]
    features = [f for f in features if f in df.columns]
    
    X = df[features]
    y = df[TARGET_COL]
    
    # Run Random 5-Fold Cross-Validation
    print("\nRunning standard Random 5-Fold Cross Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    rmse_scores = []
    r2_scores = []
    
    fold = 1
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
        
        # Apply target clipping
        train_df = df.iloc[train_idx]
        state_99th = train_df.groupby("State", observed=True)[TARGET_COL].quantile(0.99)
        ceilings = X_train["State"].map(state_99th).astype(float).fillna(15.0)
        y_train = y_train.clip(upper=ceilings)
        
        model = cb.CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            subsample=0.8,
            bootstrap_type='Bernoulli',
            random_seed=42,
            verbose=0
        )
        
        model.fit(X_train, y_train, cat_features=["State"], verbose=False)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Fold {fold} | RMSE: {rmse:.4f} | R^2: {r2:.4f}")
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        fold += 1
        
    print(f"\nAverage RMSE: {np.mean(rmse_scores):.4f}")
    print(f"Average R^2: {np.mean(r2_scores):.4f}")

if __name__ == "__main__":
    main()
