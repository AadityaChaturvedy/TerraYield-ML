"""
Generic Crop Yield Prediction Pipeline and Portability Engine.
Loads a crop config profile, processes features, runs out-of-time CV,
and fits/saves the final models.
"""
import argparse
import json
import os
import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

# ML architectures
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Core modules
from src.config import CROP_PROFILES, PROJECT_ROOT
from src.data_loader import load_and_preprocess
from src.features import build_feature_list
from src.models import get_cv_folds, clip_target_by_state, fit_models

np.random.seed(42)

def main():
    parser = argparse.ArgumentParser(description="Train regional crop yield hindcasting models.")
    parser.add_argument(
        "crop_profile",
        choices=list(CROP_PROFILES.keys()),
        help="Crop profile name defined in src/config.py (e.g., kharif_rice)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Optional path to custom crop yield CSV file."
    )
    parser.add_argument(
        "--climate_path",
        type=str,
        default=None,
        help="Optional path to custom GEE climate CSV file."
    )
    parser.add_argument(
        "--neighbors_path",
        type=str,
        default=None,
        help="Optional path to custom district neighbors JSON file."
    )
    parser.add_argument(
        "--include_district",
        action="store_true",
        default=False,
        help="Include District_mapped categorical feature."
    )
    parser.add_argument(
        "--include_ext_months",
        action="store_true",
        default=False,
        help="Include May and October extended months climate data."
    )
    parser.add_argument(
        "--include_soil_l1",
        action="store_true",
        default=False,
        help="Include Soil Moisture L1 feature."
    )
    parser.add_argument(
        "--include_interactions",
        action="store_true",
        default=False,
        help="Include climate interaction features."
    )
    parser.add_argument(
        "--include_yield_trend",
        action="store_true",
        default=False,
        help="Include yield trend feature."
    )
    parser.add_argument(
        "--include_lag3",
        action="store_true",
        default=False,
        help="Include 3-year temporal lag feature."
    )
    parser.add_argument(
        "--include_phenology",
        action="store_true",
        default=False,
        help="Include dynamic phenological alignment features."
    )
    parser.add_argument(
        "--include_sensor_flag",
        action="store_true",
        default=False,
        help="Include Post2000 sensor harmonization flag."
    )
    parser.add_argument(
        "--no_yield_lags",
        action="store_true",
        default=False,
        help="Disable all yield lag features for ablation experiment."
    )

    args = parser.parse_args()

    crop_name = args.crop_profile
    profile = CROP_PROFILES[crop_name]
    target_col = profile["target_col"]

    print(f"\n=======================================================")
    print(f" TRAINING PIPELINE FOR CROP PROFILE: {crop_name.upper()} ")
    print(f"=======================================================\n")

    print("Step 1: Loading and preprocessing data...")
    df = load_and_preprocess(
        crop_profile_name=crop_name,
        data_path=args.data_path,
        climate_path=args.climate_path,
        neighbors_path=args.neighbors_path,
        include_district_feature=args.include_district,
        include_extended_months=args.include_ext_months,
        include_soil_l1=args.include_soil_l1,
        include_interactions=args.include_interactions,
        include_yield_trend=args.include_yield_trend,
        include_lag3=args.include_lag3,
        include_phenology=args.include_phenology
    )

    features = build_feature_list(
        df,
        crop_profile_name=crop_name,
        include_district_feature=args.include_district,
        include_extended_months=args.include_ext_months,
        include_soil_l1=args.include_soil_l1,
        include_interactions=args.include_interactions,
        include_yield_trend=args.include_yield_trend,
        include_lag3=args.include_lag3,
        include_phenology=args.include_phenology,
        include_sensor_flag=args.include_sensor_flag,
        include_yield_lags=not args.no_yield_lags
    )
    
    print(f"Features engineered: {len(features)}")
    
    X = df[features]
    y = df[target_col]
    
    unique_years = np.sort(df["Year"].unique())
    print(f"Data period: {unique_years.min()} to {unique_years.max()} ({len(unique_years)} years)")

    # Out-of-time chronological CV
    print("\nStep 2: Performing Out-of-Time Chronological Cross-Validation...")
    folds = get_cv_folds(df, unique_years)
    cat_features_list = ["State"]
    if args.include_district:
        cat_features_list.append("District_mapped")

    cv_results = []
    cv_best_iters = {'xgb': [], 'lgb': [], 'cb': []}

    for fold_idx, (actual_train_years, val_years, test_years) in enumerate(folds):
        train_mask = df["Year"].isin(actual_train_years)
        val_mask = df["Year"].isin(val_years)
        test_mask = df["Year"].isin(test_years)

        X_train, y_train = X[train_mask].copy(), y[train_mask].copy()
        X_val, y_val = X[val_mask].copy(), y[val_mask].copy()
        X_test, y_test = X[test_mask].copy(), y[test_mask].copy()

        # Clip target
        y_train = clip_target_by_state(df[train_mask], X_train, y_train, target_col, profile["clipping_quantile"])

        # Train models
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

        # Collect best iterations
        cv_best_iters['xgb'].append(preds['xgb']['best_iter'])
        cv_best_iters['lgb'].append(preds['lgb']['best_iter'])
        cv_best_iters['cb'].append(preds['cb']['best_iter'])

        # Robust Blend Stacking
        p_ensemble = 0.59 * p_xgb + 0.30 * p_lgb + 0.11 * p_cb

        r2_xgb = r2_score(y_test, p_xgb)
        r2_lgb = r2_score(y_test, p_lgb)
        r2_cb = r2_score(y_test, p_cb)
        r2_ens = r2_score(y_test, p_ensemble)

        rmse_cb = np.sqrt(mean_squared_error(y_test, p_cb))
        rmse_ens = np.sqrt(mean_squared_error(y_test, p_ensemble))
        mae_ens = mean_absolute_error(y_test, p_ensemble)

        print(f"  Fold {fold_idx+1} | Test Period: {test_years.min()}-{test_years.max()} | CatBoost R²: {r2_cb:.4f} | CatBoost RMSE: {rmse_cb:.4f}")
        print(f"  Fold {fold_idx+1} | Test Period: {test_years.min()}-{test_years.max()} | Ensemble R²: {r2_ens:.4f} | Ensemble RMSE: {rmse_ens:.4f}")

        cv_results.append({
            'fold': fold_idx+1,
            'xgb_r2': r2_xgb,
            'lgb_r2': r2_lgb,
            'cb_r2': r2_cb,
            'ens_r2': r2_ens,
            'ens_rmse': rmse_ens,
            'ens_mae': mae_ens
        })

    avg_xgb_r2 = np.mean([r['xgb_r2'] for r in cv_results])
    avg_lgb_r2 = np.mean([r['lgb_r2'] for r in cv_results])
    avg_cb_r2 = np.mean([r['cb_r2'] for r in cv_results])
    avg_ens_r2 = np.mean([r['ens_r2'] for r in cv_results])
    avg_rmse = np.mean([r['ens_rmse'] for r in cv_results])
    avg_mae = np.mean([r['ens_mae'] for r in cv_results])

    print("\n" + "="*80)
    print(f"CROSS-VALIDATION PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Average XGBoost R²:  {avg_xgb_r2:.4f}")
    print(f"Average LightGBM R²: {avg_lgb_r2:.4f}")
    print(f"Average CatBoost R²: {avg_cb_r2:.4f}")
    print(f"Average Ensemble R²: {avg_ens_r2:.4f} | RMSE = {avg_rmse:.4f} | MAE = {avg_mae:.4f}")

    # Step 3: Train final models on full dataset for inference
    avg_cb_iters = int(np.round(np.mean(cv_best_iters['cb'])))
    avg_xgb_iters = int(np.round(np.mean(cv_best_iters['xgb'])))
    avg_lgb_iters = int(np.round(np.mean(cv_best_iters['lgb'])))
    
    print("\nStep 3: Training final ensemble models on the full dataset...")
    print(f"Optimal iterations found via CV (average): CatBoost={avg_cb_iters}, XGBoost={avg_xgb_iters}, LightGBM={avg_lgb_iters}")
    
    full_state_99th = df.groupby("State", observed=True)[target_col].quantile(profile["clipping_quantile"])
    ceilings = df["State"].map(full_state_99th).astype(float).fillna(15.0)
    y_final = y.clip(upper=ceilings)

    # CatBoost final model
    cb_final = cb.CatBoostRegressor(
        iterations=avg_cb_iters,
        learning_rate=profile["cb_params"]["learning_rate"],
        depth=profile["cb_params"]["depth"],
        l2_leaf_reg=profile["cb_params"]["l2_leaf_reg"],
        subsample=profile["cb_params"]["subsample"],
        bootstrap_type='Bernoulli',
        random_seed=42,
        verbose=0
    )
    cb_final.fit(X, y_final, cat_features=cat_features_list, verbose=False)

    # XGBoost final model
    xgb_final = xgb.XGBRegressor(
        n_estimators=avg_xgb_iters,
        learning_rate=profile["xgb_params"]["learning_rate"],
        max_depth=profile["xgb_params"]["max_depth"],
        subsample=profile["xgb_params"]["subsample"],
        colsample_bytree=profile["xgb_params"]["colsample_bytree"],
        reg_lambda=profile["xgb_params"]["reg_lambda"],
        reg_alpha=profile["xgb_params"]["reg_alpha"],
        enable_categorical=True,
        random_state=42
    )
    xgb_final.fit(X, y_final, verbose=False)

    # LightGBM final model
    lgb_final = lgb.LGBMRegressor(
        n_estimators=avg_lgb_iters,
        learning_rate=profile["lgb_params"]["learning_rate"],
        max_depth=profile["lgb_params"]["max_depth"],
        num_leaves=profile["lgb_params"]["num_leaves"],
        subsample=profile["lgb_params"]["subsample"],
        subsample_freq=1,
        colsample_bytree=profile["lgb_params"]["colsample_bytree"],
        reg_lambda=profile["lgb_params"]["reg_lambda"],
        reg_alpha=profile["lgb_params"]["reg_alpha"],
        random_state=42,
        verbose=-1
    )
    lgb_final.fit(X, y_final)


    # Step 4: Serialize models
    output_dir = PROJECT_ROOT / "models"
    os.makedirs(output_dir, exist_ok=True)
    
    cb_path = output_dir / f"{crop_name}_cb_model.cbm"
    xgb_path = output_dir / f"{crop_name}_xgb_model.json"
    lgb_path = output_dir / f"{crop_name}_lgb_model.txt"
    meta_path = output_dir / f"{crop_name}_ensemble_weights.json"

    cb_final.save_model(str(cb_path))
    xgb_final.save_model(str(xgb_path))
    lgb_final.booster_.save_model(str(lgb_path))

    ensemble_weights = {
        "xgb": 0.59,
        "lgb": 0.30,
        "cb": 0.11,
        "features": list(features),
        "target_col": target_col
    }
    with open(meta_path, 'w') as f:
        json.dump(ensemble_weights, f, indent=4)

    print(f"\nFinal models successfully trained and serialized to: {output_dir}/")
    print(f"Done!\n")

if __name__ == "__main__":
    main()
