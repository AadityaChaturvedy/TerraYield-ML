"""
Batch Training and Evaluation Script for All Crops.
Loops through all 15 crops defined in src/config.py, executes out-of-time
chronological validation, and compiles R2 and RMSE results.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tabulate import tabulate
import pathlib
import sys
import warnings
import json

warnings.filterwarnings('ignore')

# Add parent directory to path to allow import of src
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

from src.config import CROP_PROFILES
from src.data_loader import load_and_preprocess
from src.features import build_feature_list
from src.models import get_cv_folds, clip_target_by_state, fit_models

np.random.seed(42)

def train_and_evaluate_crop(crop_name):
    profile = CROP_PROFILES[crop_name]
    target_col = profile["target_col"]
    
    # Resolve CSV file path
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    data_path = repo_root / "data" / "processed" / f"crop_{profile['crop_name']}_season_year_wide_harmonized.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    # Load and preprocess data with the standardized baseline features
    df = load_and_preprocess(
        crop_profile_name=crop_name,
        data_path=data_path,
        include_district_feature=True,
        include_extended_months=True,
        include_soil_l1=True,
        include_interactions=False,
        include_yield_trend=False,
        include_lag3=True,
        include_phenology=False
    )
    
    # Build feature list
    features = build_feature_list(
        df,
        crop_profile_name=crop_name,
        include_district_feature=True,
        include_extended_months=True,
        include_soil_l1=True,
        include_interactions=False,
        include_yield_trend=False,
        include_lag3=True,
        include_phenology=False,
        include_sensor_flag=False
    )
    
    X = df[features]
    y = df[target_col]
    
    unique_years = np.sort(df["Year"].unique())
    folds = get_cv_folds(df, unique_years)
    
    cat_features_list = ["State", "District_mapped"]
    
    cv_results = []
    
    for fold_idx, (actual_train_years, val_years, test_years) in enumerate(folds):
        train_mask = df["Year"].isin(actual_train_years)
        val_mask = df["Year"].isin(val_years)
        test_mask = df["Year"].isin(test_years)

        X_train, y_train = X[train_mask].copy(), y[train_mask].copy()
        X_val, y_val = X[val_mask].copy(), y[val_mask].copy()
        X_test, y_test = X[test_mask].copy(), y[test_mask].copy()

        # Clip target
        y_train = clip_target_by_state(df[train_mask], X_train, y_train, target_col, profile["clipping_quantile"])

        # Train models using default config parameters
        preds = fit_models(
            X_train, y_train, X_val, y_val, X_test, y_test,
            cat_features=cat_features_list,
            cb_params=profile["cb_params"],
            xgb_params=profile["xgb_params"],
            lgb_params=profile["lgb_params"]
        )

        p_xgb = preds['xgb']['test']
        p_lgb = preds['lgb']['test']
        p_cb = preds['cb']['test']

        # Weighted blend ensemble
        p_ensemble = 0.10 * p_xgb + 0.10 * p_lgb + 0.80 * p_cb

        r2 = r2_score(y_test, p_ensemble)
        rmse = np.sqrt(mean_squared_error(y_test, p_ensemble))
        mae = mean_absolute_error(y_test, p_ensemble)

        cv_results.append({
            'fold': fold_idx+1,
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        })
        
    avg_r2 = np.mean([r['r2'] for r in cv_results])
    avg_rmse = np.mean([r['rmse'] for r in cv_results])
    avg_mae = np.mean([r['mae'] for r in cv_results])
    fold_r2s = [r['r2'] for r in cv_results]
    fold_rmses = [r['rmse'] for r in cv_results]
    
    return {
        "crop_profile": crop_name,
        "crop_name": profile["crop_name"],
        "season": profile["season_prefix"],
        "avg_r2": avg_r2,
        "avg_rmse": avg_rmse,
        "avg_mae": avg_mae,
        "fold_r2s": fold_r2s,
        "fold_rmses": fold_rmses
    }

def main():
    print("Starting batch evaluation for all 15 crops...")
    results = []
    
    for crop_name in CROP_PROFILES.keys():
        print(f"\nEvaluating: {crop_name.upper()}...")
        try:
            res = train_and_evaluate_crop(crop_name)
            results.append(res)
            print(f"  Avg R²: {res['avg_r2']:.4f} | Avg RMSE: {res['avg_rmse']:.4f}")
        except Exception as e:
            print(f"  Error evaluating {crop_name}: {e}")
            
    # Print comparison table
    table_data = []
    for res in results:
        table_data.append([
            res["crop_profile"],
            res["season"],
            f"{res['avg_r2']:.4f}",
            f"{res['avg_rmse']:.4f}",
            f"{res['avg_mae']:.4f}",
            f"{res['fold_r2s'][0]:.4f}",
            f"{res['fold_r2s'][1]:.4f}",
            f"{res['fold_r2s'][2]:.4f}"
        ])
        
    headers = ["Crop Profile", "Season", "Avg R²", "Avg RMSE", "Avg MAE", "Fold 1 R²", "Fold 2 R²", "Fold 3 R²"]
    
    print("\n" + "="*95)
    print("ALL CROPS CHRONOLOGICAL VALIDATION METRICS")
    print("="*95)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("="*95)
    
    # Save results to a json file
    output_path = pathlib.Path(__file__).parent / "all_crops_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved raw results to {output_path}")

if __name__ == "__main__":
    main()
