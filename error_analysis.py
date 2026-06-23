import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.config import CROP_PROFILES
from src.data_loader import load_and_preprocess
from src.features import build_feature_list
from src.models import get_cv_folds, fit_models, clip_target_by_state

def main():
    crop_name = "kharif_rice"
    profile = CROP_PROFILES[crop_name]
    target_col = profile["target_col"]
    
    print("Loading data and engineering features...")
    df = load_and_preprocess(
        crop_profile_name=crop_name,
        include_district_feature=True,
        include_lag3=True
    )
    features = build_feature_list(df, crop_profile_name=crop_name, include_district_feature=True, include_lag3=True)
    
    X = df[features]
    y = df[target_col]
    
    unique_years = np.sort(df["Year"].unique())
    folds = get_cv_folds(df, unique_years)
    cat_features_list = ["State", "District_mapped"]
    
    all_test_preds = []
    
    for fold_idx, (actual_train_years, val_years, test_years) in enumerate(folds):
        print(f"Processing Fold {fold_idx + 1}...")
        train_mask = df["Year"].isin(actual_train_years)
        val_mask = df["Year"].isin(val_years)
        test_mask = df["Year"].isin(test_years)

        X_train, y_train = X[train_mask].copy(), y[train_mask].copy()
        y_train = clip_target_by_state(df[train_mask], X_train, y_train, target_col, profile["clipping_quantile"])

        X_val, y_val = X[val_mask].copy(), y[val_mask].copy()
        X_test, y_test = X[test_mask].copy(), y[test_mask].copy()
        
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
        p_ens = 0.10 * p_xgb + 0.10 * p_lgb + 0.80 * p_cb
        
        fold_df = df[test_mask][['State', 'District_mapped', 'Year', target_col]].copy()
        fold_df['Pred'] = p_ens
        fold_df['Fold'] = fold_idx + 1
        all_test_preds.append(fold_df)
        
    res = pd.concat(all_test_preds, axis=0)
    
    # Analysis 1: By State
    state_metrics = []
    for state, group in res.groupby('State'):
        if len(group) < 10: continue
        r2 = r2_score(group[target_col], group['Pred'])
        mae = mean_absolute_error(group[target_col], group['Pred'])
        rmse = np.sqrt(mean_squared_error(group[target_col], group['Pred']))
        state_metrics.append({'State': state, 'Count': len(group), 'R2': r2, 'MAE': mae, 'RMSE': rmse, 'Mean_Yield': group[target_col].mean()})
    
    state_df = pd.DataFrame(state_metrics).sort_values('R2', ascending=False)
    print("\n--- PERFORMANCE BY STATE ---")
    print(state_df.to_string(index=False))
    
    # Analysis 2: High Yield vs Low Yield
    # Let's define the median historical yield of a district to classify them
    district_means = df.groupby('District_mapped')[target_col].mean().reset_index()
    district_means.rename(columns={target_col: 'Hist_Mean_Yield'}, inplace=True)
    res = res.merge(district_means, on='District_mapped', how='left')
    
    median_yield = res['Hist_Mean_Yield'].median()
    res['Yield_Regime'] = np.where(res['Hist_Mean_Yield'] >= median_yield, 'High Yield', 'Low Yield')
    
    regime_metrics = []
    for regime, group in res.groupby('Yield_Regime'):
        r2 = r2_score(group[target_col], group['Pred'])
        mae = mean_absolute_error(group[target_col], group['Pred'])
        rmse = np.sqrt(mean_squared_error(group[target_col], group['Pred']))
        regime_metrics.append({'Regime': regime, 'Count': len(group), 'R2': r2, 'MAE': mae, 'RMSE': rmse})
        
    print("\n--- PERFORMANCE BY YIELD REGIME ---")
    print(pd.DataFrame(regime_metrics).to_string(index=False))
    
    # Analysis 3: By Year (Drought years?)
    year_metrics = []
    for yr, group in res.groupby('Year'):
        r2 = r2_score(group[target_col], group['Pred'])
        mae = mean_absolute_error(group[target_col], group['Pred'])
        year_metrics.append({'Year': yr, 'Count': len(group), 'R2': r2, 'MAE': mae})
        
    print("\n--- PERFORMANCE BY YEAR ---")
    print(pd.DataFrame(year_metrics).to_string(index=False))

if __name__ == "__main__":
    main()
