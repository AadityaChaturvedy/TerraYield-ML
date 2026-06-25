# Changelog

All notable changes to the **TerraYield-ML (TERRA)** project will be documented in this file.

## [1.0.0] - 2026-06-23

### Added
- **Core Modeling Pipeline**: Standardized CLI (`train_crop.py`) supporting both the primary interpretable **XGBoost** model and a **Stacked Ensemble** (CatBoost + XGBoost + LightGBM).
- **Out-of-Time Chronological Validation**: Enforces temporal splits to eliminate data leakage and simulate true operational hindcasting.
- **Model Interpretability (SHAP)**: Added `shap_analysis.py` to generate and save SHAP beeswarm and feature importance plots.
- **Cropland Masking in GEE**: Integrated MODIS `MCD12Q1` cropland classification masking into the Earth Engine weather data aggregator (`gee_climate_extraction.py`).
- **Ensemble Weight Optimization**: Added `scratch/optimize_ensemble_weights.py` to run grid sweeps on out-of-fold validation predictions.
- **Imputation Validation**: Added `scratch/validate_ndvi_imputation.py` to validate pre-2000 MODIS climatology reconstruction against actual observed satellite pixels.
- **Version Pinning**: Staged exact versions in `requirements.txt` to prevent dependency breakage.

### Changed
- **Target Refit Logic**: Modified the final model fitting steps to use the average best iterations computed during chronological cross-validation folds instead of static values.
- **Unused Code Removal**: Integrated and utilized `cv_best_iters` in `train_crop.py` for model final training.
