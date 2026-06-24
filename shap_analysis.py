"""
Model Interpretability and Feature Attribution via SHAP.
Trains the primary XGBoost model on the selected crop (default: kharif_rice)
and generates global interpretability plots (beeswarm and feature importance).
"""
import argparse
import os
import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path to allow import of src
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

from src.config import CROP_PROFILES, PROJECT_ROOT
from src.data_loader import load_and_preprocess
from src.features import build_feature_list
from src.models import clip_target_by_state

np.random.seed(42)

def run_shap_analysis(crop_name, output_dir):
    profile = CROP_PROFILES[crop_name]
    target_col = profile["target_col"]
    
    # Resolve CSV file path
    data_path = PROJECT_ROOT / "data" / "processed" / f"crop_{profile['crop_name']}_season_year_wide_harmonized.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    print(f"Loading and preprocessing data for {crop_name}...")
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
    
    X = df[features].copy()
    y = df[target_col].copy()
    
    # Ensure object columns are cast to category for XGBoost categorical feature support
    cat_cols = ["State", "District_mapped"]
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')
            
    # Clip training target using state-specific 99th percentile (same as pipeline)
    y_clipped = clip_target_by_state(df, X, y, target_col, profile["clipping_quantile"])
    
    print("Training primary XGBoost model on full dataset...")
    # Initialize XGBoost with optimized hyperparameters from config
    xgb_params = profile["xgb_params"].copy()
    xgb_params.pop("enable_categorical", None) # Ensure compatibility
    
    model = xgb.XGBRegressor(
        n_estimators=xgb_params.get("n_estimators", 500),
        learning_rate=xgb_params.get("learning_rate", 0.05),
        max_depth=xgb_params.get("max_depth", 6),
        subsample=xgb_params.get("subsample", 0.8),
        colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
        reg_lambda=xgb_params.get("reg_lambda", 5.0),
        reg_alpha=xgb_params.get("reg_alpha", 1.0),
        enable_categorical=True,
        random_state=42
    )
    model.fit(X, y_clipped, verbose=False)
    
    print("Calculating SHAP values using TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. SHAP Beeswarm Plot
    print(f"Generating SHAP beeswarm plot -> {output_dir}/shap_beeswarm.png")
    plt.figure(figsize=(12, 8))
    # Note: shap.plots.beeswarm doesn't support passing plt ax directly, but modifies active figure
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title(f"SHAP Feature Attribution: {crop_name.upper()} Yield Model", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. SHAP Bar Feature Importance Plot
    print(f"Generating SHAP feature importance plot -> {output_dir}/shap_importance.png")
    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title(f"Mean Absolute SHAP Feature Importance: {crop_name.upper()} Yield Model", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nSHAP analysis successfully completed! Output plots saved to:")
    print(f"  - {output_dir / 'shap_beeswarm.png'}")
    print(f"  - {output_dir / 'shap_importance.png'}")

def main():
    parser = argparse.ArgumentParser(description="Generate SHAP interpretability plots for crop yield models.")
    parser.add_argument(
        "--crop",
        type=str,
        default="kharif_rice",
        choices=list(CROP_PROFILES.keys()),
        help="Crop profile name (default: kharif_rice)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory to save output plots (default: plots)"
    )
    args = parser.parse_args()
    
    output_path = pathlib.Path(args.output_dir)
    
    try:
        run_shap_analysis(args.crop, output_path)
    except Exception as e:
        print(f"Error during SHAP analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
