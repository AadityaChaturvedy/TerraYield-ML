"""
Evaluate adding dynamic phenology-aligned features and sensor boundary flags
to the Kharif Rice model under out-of-time chronological validation.
"""
import numpy as np
import pandas as pd
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pathlib
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to allow import of src
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

from src.config import CROP_PROFILES
from src.data_loader import load_and_preprocess
from src.features import build_feature_list
from src.models import get_cv_folds, clip_target_by_state, fit_models

np.random.seed(42)

def main():
    crop_name = "kharif_rice"
    profile = CROP_PROFILES[crop_name]
    target_col = profile["target_col"]

    print("Loading data with new data loader...")
    # This will load standard May-Oct climate and compute phenology-aligned features
    df = load_and_preprocess(
        crop_profile_name=crop_name,
        include_district_feature=True,
        include_extended_months=True,
        include_soil_l1=True,
        include_interactions=False,
        include_yield_trend=False,
        include_lag3=True,
        include_phenology=True,
        include_sensor_flag=True
    )
    
    # Check if phenology features were created
    aligned_cols = [c for c in df.columns if "_Peak" in c]
    print(f"Number of phenology-aligned features generated: {len(aligned_cols)}")
    print(f"Sample phenology columns: {aligned_cols[:10]}")
    
    # 1. Build features WITHOUT phenology-aligned columns (Base)
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
    # Remove aligned cols and Post2000 from features_base for a fair comparison
    features_base = [f for f in features_base if f not in aligned_cols and f != "Post2000"]
    
    # 2. Build features WITH phenology-aligned columns and Post2000 (New)
    features_new = build_feature_list(
        df,
        crop_profile_name=crop_name,
        include_district_feature=True,
        include_extended_months=True,
        include_soil_l1=True,
        include_interactions=False,
        include_yield_trend=False,
        include_lag3=True,
        include_phenology=True,
        include_sensor_flag=True
    )
    
    print(f"Base Feature count: {len(features_base)}")
    print(f"New Feature count: {len(features_new)} (including Post2000 and phenology features)")

    X_base = df[features_base]
    X_new = df[features_new]
    y = df[target_col]
    
    unique_years = np.sort(df["Year"].unique())
    folds = get_cv_folds(df, unique_years)

    cat_features_list = ["State", "District_mapped"]
    
    # CV loops
    print("\nRunning Out-of-Time Chronological CV for Base Model...")
    base_results = []
    for fold_idx, (actual_train_years, val_years, test_years) in enumerate(folds):
        train_mask = df["Year"].isin(actual_train_years)
        val_mask = df["Year"].isin(val_years)
        test_mask = df["Year"].isin(test_years)

        X_train, y_train = X_base[train_mask].copy(), y[train_mask].copy()
        X_val, y_val = X_base[val_mask].copy(), y[val_mask].copy()
        X_test, y_test = X_base[test_mask].copy(), y[test_mask].copy()
        
        y_train = clip_target_by_state(df[train_mask], X_train, y_train, target_col, profile["clipping_quantile"])

        preds = fit_models(
            X_train, y_train, X_val, y_val, X_test, y_test,
            cat_features=cat_features_list,
            cb_params=profile["cb_params"],
            xgb_params=profile["xgb_params"],
            lgb_params=profile["lgb_params"]
        )
        
        p_ens = 0.10 * preds['xgb']['test'] + 0.10 * preds['lgb']['test'] + 0.80 * preds['cb']['test']
        r2_ens = r2_score(y_test, p_ens)
        rmse_ens = np.sqrt(mean_squared_error(y_test, p_ens))
        
        print(f"  Fold {fold_idx+1} ({test_years.min()}-{test_years.max()}): R² = {r2_ens:.4f} | RMSE = {rmse_ens:.4f}")
        base_results.append({'r2': r2_ens, 'rmse': rmse_ens})

    print("\nRunning Out-of-Time Chronological CV for New Model (with Phenology & Post2000)...")
    new_results = []
    for fold_idx, (actual_train_years, val_years, test_years) in enumerate(folds):
        train_mask = df["Year"].isin(actual_train_years)
        val_mask = df["Year"].isin(val_years)
        test_mask = df["Year"].isin(test_years)

        X_train, y_train = X_new[train_mask].copy(), y[train_mask].copy()
        X_val, y_val = X_new[val_mask].copy(), y[val_mask].copy()
        X_test, y_test = X_new[test_mask].copy(), y[test_mask].copy()
        
        y_train = clip_target_by_state(df[train_mask], X_train, y_train, target_col, profile["clipping_quantile"])

        preds = fit_models(
            X_train, y_train, X_val, y_val, X_test, y_test,
            cat_features=cat_features_list,
            cb_params=profile["cb_params"],
            xgb_params=profile["xgb_params"],
            lgb_params=profile["lgb_params"]
        )
        
        p_ens = 0.10 * preds['xgb']['test'] + 0.10 * preds['lgb']['test'] + 0.80 * preds['cb']['test']
        r2_ens = r2_score(y_test, p_ens)
        rmse_ens = np.sqrt(mean_squared_error(y_test, p_ens))
        
        print(f"  Fold {fold_idx+1} ({test_years.min()}-{test_years.max()}): R² = {r2_ens:.4f} | RMSE = {rmse_ens:.4f}")
        new_results.append({'r2': r2_ens, 'rmse': rmse_ens})

    # Summary
    avg_r2_base = np.mean([r['r2'] for r in base_results])
    avg_rmse_base = np.mean([r['rmse'] for r in base_results])
    avg_r2_new = np.mean([r['r2'] for r in new_results])
    avg_rmse_new = np.mean([r['rmse'] for r in new_results])

    print("\n" + "="*80)
    print("PHENOLOGICAL ALIGNMENT EVALUATION SUMMARY")
    print("="*80)
    print(f"Average Base Model: R² = {avg_r2_base:.4f} | RMSE = {avg_rmse_base:.4f}")
    print(f"Average New Model:  R² = {avg_r2_new:.4f} | RMSE = {avg_rmse_new:.4f}")
    print(f"Improvement:        Delta R² = {avg_r2_new - avg_r2_base:+.4f} | Delta RMSE = {avg_rmse_new - avg_rmse_base:+.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
