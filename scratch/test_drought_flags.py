"""
Evaluate adding binary severe weather flags to help the model override 
the lag dominance during extreme years (e.g. Jharkhand 2022).
"""
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.metrics import r2_score, mean_squared_error
import pathlib
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to allow import of src
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

from src.config import CROP_PROFILES
from src.data_loader import load_and_preprocess
from src.features import build_feature_list
from src.models import get_cv_folds, clip_target_by_state

np.random.seed(42)

def main():
    crop_name = "kharif_rice"
    profile = CROP_PROFILES[crop_name]
    target_col = profile["target_col"]

    print("Loading data...")
    df = load_and_preprocess(
        crop_profile_name=crop_name,
        include_district_feature=True,
        include_extended_months=True,
        include_soil_l1=True,
        include_interactions=False,
        include_yield_trend=False,
        include_lag3=True
    )
    
    # Add binary severe weather indicators
    month_names = ["Jun", "Jul", "Aug", "Sep"]
    for m in month_names:
        # Drought flag: precip or soil moisture more than 1.2 std dev below normal
        p_z = df[f"Precip_{m}_Z"]
        s_z = df[f"Soil_{m}_Z"]
        df[f"Is_Drought_{m}"] = ((p_z < -1.2) | (s_z < -1.2)).astype(int)
        
        # Heat stress flag: temperature more than 1.2 std dev above normal
        t_z = df[f"Temp_{m}_Z"]
        df[f"Is_Heatwave_{m}"] = (t_z > 1.2).astype(int)

    features_base = build_feature_list(
        df,
        crop_profile_name=crop_name,
        include_district_feature=True,
        include_extended_months=True,
        include_soil_l1=True,
        include_interactions=False,
        include_yield_trend=False,
        include_lag3=True
    )
    
    # Add new flags to feature set
    flag_features = []
    for m in month_names:
        flag_features.append(f"Is_Drought_{m}")
        flag_features.append(f"Is_Heatwave_{m}")
        
    features_with_flags = features_base + flag_features

    X_base = df[features_base]
    X_flags = df[features_with_flags]
    y = df[target_col]
    
    unique_years = np.sort(df["Year"].unique())
    folds = get_cv_folds(df, unique_years)

    cat_features_list = ["State", "District_mapped"]
    cb_params = profile["cb_params"]

    print("\nEvaluating CatBoost with and without Severe Weather Flags...")
    all_results = []

    for fold_idx, (actual_train_years, val_years, test_years) in enumerate(folds):
        train_mask = df["Year"].isin(actual_train_years)
        val_mask = df["Year"].isin(val_years)
        test_mask = df["Year"].isin(test_years)

        # 1. Base model
        X_train_base, y_train_base = X_base[train_mask].copy(), y[train_mask].copy()
        X_val_base, y_val_base = X_base[val_mask].copy(), y[val_mask].copy()
        X_test_base, y_test_base = X_base[test_mask].copy(), y[test_mask].copy()
        
        y_train_base = clip_target_by_state(df[train_mask], X_train_base, y_train_base, target_col, profile["clipping_quantile"])

        model_base = cb.CatBoostRegressor(**cb_params, early_stopping_rounds=30)
        model_base.fit(X_train_base, y_train_base, cat_features=cat_features_list, eval_set=(X_val_base, y_val_base), verbose=False)
        p_base = model_base.predict(X_test_base)

        # 2. Flag model
        X_train_flag, y_train_flag = X_flags[train_mask].copy(), y[train_mask].copy()
        X_val_flag, y_val_flag = X_flags[val_mask].copy(), y[val_mask].copy()
        X_test_flag, y_test_flag = X_flags[test_mask].copy(), y[test_mask].copy()
        
        y_train_flag = clip_target_by_state(df[train_mask], X_train_flag, y_train_flag, target_col, profile["clipping_quantile"])

        model_flag = cb.CatBoostRegressor(**cb_params, early_stopping_rounds=30)
        model_flag.fit(X_train_flag, y_train_flag, cat_features=cat_features_list, eval_set=(X_val_flag, y_val_flag), verbose=False)
        p_flag = model_flag.predict(X_test_flag)

        r2_base = r2_score(y_test_base, p_base)
        r2_flag = r2_score(y_test_base, p_flag)

        rmse_base = np.sqrt(mean_squared_error(y_test_base, p_base))
        rmse_flag = np.sqrt(mean_squared_error(y_test_base, p_flag))

        print(f"Fold {fold_idx+1} ({test_years.min()}-{test_years.max()}):")
        print(f"  Base CatBoost: R² = {r2_base:.4f} | RMSE = {rmse_base:.4f}")
        print(f"  Flag CatBoost: R² = {r2_flag:.4f} | RMSE = {rmse_flag:.4f}")

        all_results.append({
            'fold': fold_idx+1,
            'r2_base': r2_base, 'r2_flag': r2_flag,
            'rmse_base': rmse_base, 'rmse_flag': rmse_flag
        })

    avg_r2_base = np.mean([r['r2_base'] for r in all_results])
    avg_r2_flag = np.mean([r['r2_flag'] for r in all_results])
    avg_rmse_base = np.mean([r['rmse_base'] for r in all_results])
    avg_rmse_flag = np.mean([r['rmse_flag'] for r in all_results])

    print("\n" + "="*80)
    print("SEVERE WEATHER FLAGS BENCHMARK SUMMARY")
    print("="*80)
    print(f"Average Base CatBoost: R² = {avg_r2_base:.4f} | RMSE = {avg_rmse_base:.4f}")
    print(f"Average Flag CatBoost: R² = {avg_r2_flag:.4f} | RMSE = {avg_rmse_flag:.4f}")

if __name__ == "__main__":
    main()
