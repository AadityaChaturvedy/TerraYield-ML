"""
Optuna hyperparameter optimization script for Kharif Rice.
Runs focused searches on XGBoost, LightGBM, and CatBoost
using chronological time-series cross-validation.
"""
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

from src.config import CROP_PROFILES
from src.data_loader import load_and_preprocess
from src.features import build_feature_list
from src.models import get_cv_folds, clip_target_by_state

optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)

def main():
    crop_name = "kharif_rice"
    profile = CROP_PROFILES[crop_name]
    target_col = profile["target_col"]

    print("Loading data for tuning...")
    df = load_and_preprocess(
        crop_profile_name=crop_name,
        include_district_feature=True,
        include_extended_months=True,
        include_soil_l1=True,
        include_interactions=False,
        include_yield_trend=False,
        include_lag3=True
    )
    
    features = build_feature_list(
        df,
        crop_profile_name=crop_name,
        include_district_feature=True,
        include_extended_months=True,
        include_soil_l1=True,
        include_interactions=False,
        include_yield_trend=False,
        include_lag3=True
    )

    X = df[features]
    y = df[target_col]
    unique_years = np.sort(df["Year"].unique())
    folds = get_cv_folds(df, unique_years)

    cat_features_list = ["State", "District_mapped"]

    # 1. OPTUNA: CatBoost (30 trials)
    print("\n>>> Tuning CatBoost (30 trials)...")
    def objective_cb(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 40.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.1, 8.0, log=True),
            'bootstrap_type': 'Bernoulli',
            'random_seed': 42,
            'verbose': 0,
            'early_stopping_rounds': 30
        }
        
        fold_r2s = []
        for actual_train_years, val_years, test_years in folds:
            train_mask = df["Year"].isin(actual_train_years)
            val_mask = df["Year"].isin(val_years)
            test_mask = df["Year"].isin(test_years)
            
            X_train, y_train = X[train_mask].copy(), y[train_mask].copy()
            X_val, y_val = X[val_mask].copy(), y[val_mask].copy()
            X_test, y_test = X[test_mask].copy(), y[test_mask].copy()
            
            y_train = clip_target_by_state(df[train_mask], X_train, y_train, target_col, profile["clipping_quantile"])
            
            model = cb.CatBoostRegressor(**params)
            model.fit(X_train, y_train, cat_features=cat_features_list, eval_set=(X_val, y_val), verbose=False)
            p = model.predict(X_test)
            fold_r2s.append(r2_score(y_test, p))
            
        return np.mean(fold_r2s)

    study_cb = optuna.create_study(direction='maximize')
    study_cb.optimize(objective_cb, n_trials=30)
    best_cb = study_cb.best_params
    print(f"  Best CatBoost R²: {study_cb.best_value:.4f}")
    print(f"  Best parameters: {best_cb}")

    # 2. OPTUNA: XGBoost (20 trials)
    print("\n>>> Tuning XGBoost (20 trials)...")
    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 40.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0, log=True),
            'enable_categorical': True,
            'random_state': 42,
            'early_stopping_rounds': 30
        }
        
        fold_r2s = []
        for actual_train_years, val_years, test_years in folds:
            train_mask = df["Year"].isin(actual_train_years)
            val_mask = df["Year"].isin(val_years)
            test_mask = df["Year"].isin(test_years)
            
            X_train, y_train = X[train_mask].copy(), y[train_mask].copy()
            X_val, y_val = X[val_mask].copy(), y[val_mask].copy()
            X_test, y_test = X[test_mask].copy(), y[test_mask].copy()
            
            y_train = clip_target_by_state(df[train_mask], X_train, y_train, target_col, profile["clipping_quantile"])
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            p = model.predict(X_test)
            fold_r2s.append(r2_score(y_test, p))
            
        return np.mean(fold_r2s)

    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=20)
    best_xgb = study_xgb.best_params
    print(f"  Best XGBoost R²: {study_xgb.best_value:.4f}")
    print(f"  Best parameters: {best_xgb}")

    # 3. OPTUNA: LightGBM (20 trials)
    print("\n>>> Tuning LightGBM (20 trials)...")
    def objective_lgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'num_leaves': trial.suggest_int('num_leaves', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'subsample_freq': 1,
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 40.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0, log=True),
            'random_state': 42,
            'verbose': -1
        }
        
        fold_r2s = []
        for actual_train_years, val_years, test_years in folds:
            train_mask = df["Year"].isin(actual_train_years)
            val_mask = df["Year"].isin(val_years)
            test_mask = df["Year"].isin(test_years)
            
            X_train, y_train = X[train_mask].copy(), y[train_mask].copy()
            X_val, y_val = X[val_mask].copy(), y[val_mask].copy()
            X_test, y_test = X[test_mask].copy(), y[test_mask].copy()
            
            y_train = clip_target_by_state(df[train_mask], X_train, y_train, target_col, profile["clipping_quantile"])
            
            model = lgb.LGBMRegressor(**params)
            callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
            p = model.predict(X_test)
            fold_r2s.append(r2_score(y_test, p))
            
        return np.mean(fold_r2s)

    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=20)
    best_lgb = study_lgb.best_params
    print(f"  Best LightGBM R²: {study_lgb.best_value:.4f}")
    print(f"  Best parameters: {best_lgb}")

    print("\n>>> Tuning Complete! Best Parameter Profiles:\n")
    print(f"CB_BEST_PARAMS = {best_cb}")
    print(f"XGB_BEST_PARAMS = {best_xgb}")
    print(f"LGB_BEST_PARAMS = {best_lgb}")

if __name__ == "__main__":
    main()
