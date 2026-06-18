"""
Modular Feature Ablation Study Runner.
Runs the out-of-time chronological cross-validation for each feature stage
on Kharif Rice and prints a comparison table.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tabulate import tabulate
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

def run_ablation_stage(crop_name, profile, stage_opts):
    target_col = profile["target_col"]
    
    # Load and preprocess data
    df = load_and_preprocess(
        crop_profile_name=crop_name,
        include_district_feature=stage_opts["include_district_feature"],
        include_extended_months=stage_opts["include_extended_months"],
        include_soil_l1=stage_opts["include_soil_l1"],
        include_interactions=stage_opts["include_interactions"],
        include_yield_trend=stage_opts["include_yield_trend"],
        include_lag3=stage_opts["include_lag3"]
    )
    
    # Build features
    features = build_feature_list(
        df,
        crop_profile_name=crop_name,
        include_district_feature=stage_opts["include_district_feature"],
        include_extended_months=stage_opts["include_extended_months"],
        include_soil_l1=stage_opts["include_soil_l1"],
        include_interactions=stage_opts["include_interactions"],
        include_yield_trend=stage_opts["include_yield_trend"],
        include_lag3=stage_opts["include_lag3"]
    )
    
    X = df[features]
    y = df[target_col]
    
    unique_years = np.sort(df["Year"].unique())
    folds = get_cv_folds(df, unique_years)
    
    cat_features_list = ["State"]
    if stage_opts["include_district_feature"]:
        cat_features_list.append("District_mapped")
        
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
    
    return avg_r2, avg_rmse, avg_mae, [r['r2'] for r in cv_results]

def main():
    crop_name = "kharif_rice"
    profile = CROP_PROFILES[crop_name]
    
    stages = [
        {
            "name": "Stage 0: Baseline (Jun-Sep Climate)",
            "opts": {
                "include_district_feature": False,
                "include_extended_months": False,
                "include_soil_l1": False,
                "include_interactions": False,
                "include_yield_trend": False,
                "include_lag3": False
            }
        },
        {
            "name": "Stage 1: + District Mapping",
            "opts": {
                "include_district_feature": True,
                "include_extended_months": False,
                "include_soil_l1": False,
                "include_interactions": False,
                "include_yield_trend": False,
                "include_lag3": False
            }
        },
        {
            "name": "Stage 2: + Extended Months (May/Oct)",
            "opts": {
                "include_district_feature": True,
                "include_extended_months": True,
                "include_soil_l1": False,
                "include_interactions": False,
                "include_yield_trend": False,
                "include_lag3": False
            }
        },
        {
            "name": "Stage 3: + Soil Moisture L1",
            "opts": {
                "include_district_feature": True,
                "include_extended_months": True,
                "include_soil_l1": True,
                "include_interactions": False,
                "include_yield_trend": False,
                "include_lag3": False
            }
        },
        {
            "name": "Stage 4: + Interaction Features",
            "opts": {
                "include_district_feature": True,
                "include_extended_months": True,
                "include_soil_l1": True,
                "include_interactions": True,
                "include_yield_trend": False,
                "include_lag3": False
            }
        },
        {
            "name": "Stage 5: + Yield Trend Slope",
            "opts": {
                "include_district_feature": True,
                "include_extended_months": True,
                "include_soil_l1": True,
                "include_interactions": True,
                "include_yield_trend": True,
                "include_lag3": False
            }
        }
    ]

    print(f"\n=======================================================")
    print(f" RUNNING INCREMENTAL ABLATION STUDY FOR: {crop_name.upper()} ")
    print(f"=======================================================\n")
    
    results_table = []
    
    for stage in stages:
        name = stage["name"]
        print(f"Running {name}...")
        avg_r2, avg_rmse, avg_mae, fold_r2s = run_ablation_stage(crop_name, profile, stage["opts"])
        results_table.append([
            name,
            f"{avg_r2:.4f}",
            f"{avg_rmse:.4f}",
            f"{avg_mae:.4f}",
            f"{fold_r2s[0]:.4f}",
            f"{fold_r2s[1]:.4f}",
            f"{fold_r2s[2]:.4f}"
        ])
        print(f"  Result -> Avg R²: {avg_r2:.4f} | Avg RMSE: {avg_rmse:.4f}\n")

    headers = ["Ablation Stage", "Avg R²", "Avg RMSE", "Avg MAE", "Fold 1 R²", "Fold 2 R²", "Fold 3 R²"]
    print("\n" + "="*80)
    print("ABLATION STUDY BENCHMARK SUMMARY")
    print("="*80)
    print(tabulate(results_table, headers=headers, tablefmt="grid"))
    print("="*80)

if __name__ == "__main__":
    main()
