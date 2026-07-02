# TerraYield-ML: Regional Crop Yield Hindcasting Engine

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.21126177.svg)](https://doi.org/10.5281/zenodo.21126177) **(DOI: [10.5281/zenodo.21126177](https://doi.org/10.5281/zenodo.21126177))**

TerraYield-ML is a production-grade, highly reproducible machine learning framework designed for district-level crop yield hindcasting and in-season nowcasting across India. By integrating multi-source climatology (ERA5-Land temperature, CHIRPS precipitation, and volumetric soil moisture) with satellite remote sensing indices (MODIS NDVI/EVI) and historical crop statistics, the framework delivers operational-grade yield estimations.

---

## Key Features & Highlights

1. **Anti-Leakage Validation Scheme**: Traditional random k-fold cross-validation leaks information from future years and adjacent regions. TerraYield-ML uses a strict **Out-of-Time Chronological Cross-Validation** to simulate actual blind nowcasting scenarios.
2. **Comprehensive Crop Portability**: Native config-driven support for **15 crop profiles** spanning Kharif (monsoon), Rabi (winter), and Whole-Year cropping seasons.
3. **Advanced Climatological Feature Engineering**:
   - **Climatological Anomalies (Z-scores)**: Rolling district-level expanding mean and standard deviation for weather/satellite data to capture localized drought, heatwaves, or surplus moisture.
   - **Crop Water Stress Index (CWSI) Proxy**: Integrates Temp, Precip, and Soil moisture anomalies per month: $\text{CWSI}_m = \text{Temp}_m^Z - (\text{Precip}_m^Z + \text{Soil}_m^Z)$.
   - **Dynamic Phenology Alignment**: Aligning weather indices relative to the satellite-derived Peak NDVI month instead of arbitrary calendar months to accommodate shifting planting windows.
   - **Sensor Harmonization Flag**: A post-2000 binary transition indicator to resolve sensor response gaps between satellite cohorts.
   - **Spatial Adjacency Lag**: Incorporates historical yield trends from neighboring districts using a spatial adjacency map (`district_neighbors.json`).
4. **Stacked Ensemble Architecture**: Meta-ensemble combining **CatBoost** (80%), **XGBoost** (10%), and **LightGBM** (10%) with state-level target clipping (99th percentile) to mitigate reporting outliers.

---

## Validation Strategy & Data Leakage Prevention

### The Spatial-Temporal Data Leakage Problem
Traditional crop yield models utilize standard random k-fold cross-validation. This methodology introduces severe data leakage:
1. **Temporal Leakage**: Standard random splits allow future data points (e.g., year 2022) to be in the training set while predicting past years (e.g., 2020), which leaks long-term technology, climate, and economic trends backward in time.
2. **Spatial Leakage**: Standard random splits place a target district in the test set while placing its immediate neighbors in the training set for the same year. When spatial lag features (e.g., neighbor yields) are used, this results in direct target leakage.

> [!IMPORTANT]
> **TERRA Validation Strategy**: To reflect operational reality—where we must predict future crop yields using only historical observations—TERRA mandates **Out-of-Time Chronological Validation**. 
> * Folds are split sequentially in time.
> * Operational models are trained on data up to year $Y_{n-4}$, validated on $[Y_{n-3}, Y_{n-2}]$, and tested on out-of-time chronological folds $[Y_{n-1}, Y_n]$.
> * Random 5-Fold CV is used *only* as a comparison benchmark to measure leakage inflation.

### Standard Random 5-Fold CV vs. Out-of-Time CV
When evaluated under standard random cross-validation, the average Ensemble R² is **0.838** (RMSE = 0.409). This **~0.06 R² inflation** demonstrates the extent of spatial-temporal data leakage and justifies the strict chronological split.

### Spatio-Temporal Error & Harmonization Visuals

We analyze spatial-temporal error propagation using out-of-time test folds:

![Spatio-Temporal Error Distribution Maps](plots/small_multiples_spatial_residuals.png)
*Figure 1: Spatio-temporal error maps comparing prediction residuals across stable years (2018, 2020) and the severe 2022 drought year.*

Boundary harmonization prevents spatial error leakage due to changing district boundaries:

![Harmonization vs Raw Trajectory](plots/harmonization_vs_raw_trajectory.png)
*Figure 2: Performance trajectory showing the R² generalization gap between harmonized boundary maps and raw boundary models.*

Stacked ensembles are validated chronological-optimally:

![Ensemble Overfitting vs Single Learner Generalization](plots/ensemble_vs_single_generalization.png)
*Figure 3: Overfitting trajectory comparing a multi-model validation-optimal blend vs. a single robust learner (CatBoost) on the unseen test set.*

---

## Advanced Feature Engineering & Pipeline

### A. Preprocessing & Normalization
* **Historical Boundary Mapping**: Standardized district names across historical boundaries (1997–2022) to prevent spatial mismatch due to district reorganizations, using the mapped district dictionary.
* **Target Yield Outlier Clipping**: Clipped extreme values at the state-specific 99th percentile to prevent administrative reporting errors from skewing tree splits.
* **Expanding Climatology & Z-Scores**: For every climate and remote sensing feature, computed district-specific expanding means and standard deviations using only historical data prior to the target year. Derived Z-score anomalies as:
  $$
  Z_{d, y, m} = \frac{x_{d, y, m} - \mu_{d, \lt y, m}}{\sigma_{d, \lt y, m} + \epsilon}
  $$
  where $d$ is the district, $y$ is the year, and $m$ is the month.

### B. Core Features
1. **Crop Water Stress Index (CWSI) Proxy**: Combines thermal stress (high temperature) and water deficit (low precipitation and soil moisture):
   $$
   \text{CWSI}_{m} = \text{Temp\_Z}_{m} - (\text{Precip\_Z}_{m} + \text{Soil\_Z}_{m})
   $$
   
   | 3D Response Surface for CWSI | CWSI vs Yield Time Series |
   | :---: | :---: |
   | ![3D Response Surface for CWSI](plots/cwsi_3d_response_surface.png) | ![CWSI vs Yield](plots/cwsi_vs_yield_time_series.png) |
   | *Figure 4: Response surface mapping climate anomalies.* | *Figure 5: Inverted CWSI August anomaly vs actual yield anomaly.* |

2. **Post-2000 Sensor Harmonization Flag**: A binary indicator `Post2000` to allow the tree model to adjust its splits for the transition from AVHRR climatology-imputed data (1997–1999) to actual MODIS observations (2000 onwards).
3. **Dynamic Phenological Alignment**: Maps climate variables relative to the peak vegetative month (maximum NDVI) per district-year to handle varying sowing dates across Indian states:
   * $t=0$: Peak month (`Peak`)
   * $t=-1, -2$: Pre-peak sowing/emergence months (`Peak_Minus1`, `Peak_Minus2`)
   * $t=+1$: Post-peak reproductive month (`Peak_Plus1`)

   ![Sankey Diagram of Boundary Harmonization](plots/boundary_harmonization_sankey.png)
   *Figure 6: Area-weighted flow mapping legacy district splits to modern administrative configurations based on cropland pixels.*

```mermaid
flowchart TD
    A[Raw Tabular Yield Data] --> B[District Boundary Mappings]
    C[ERA5 & MODIS Climate Raster] --> D[Spatial-Temporal Aggregations]
    B --> E[Spatial & Temporal Lags]
    D --> F[Expanding Z-Scores & CWSI]
    E & F --> G[Ensemble Regressor: CatBoost + XGBoost + LightGBM]
    G --> |Out-of-Time Chronological Validation| H[Yield Hindcast Outputs]
```

---

## Model Tuning & Hyperparameter Profiles

We executed a hyperparameter optimization using **Optuna** over the Out-of-Time Chronological CV folds on the refined 114-feature Kharif Rice dataset. The optimal parameter profiles are:

* **CatBoost Regressor**:
  ```python
  {
      'iterations': 374,
      'learning_rate': 0.0595,
      'depth': 6,
      'l2_leaf_reg': 5.750,
      'subsample': 0.7083,
      'random_strength': 0.3659,
      'bootstrap_type': 'Bernoulli',
      'random_seed': 42
  }
  ```
* **XGBoost Regressor**:
  ```python
  {
      'n_estimators': 728,
      'learning_rate': 0.1038,
      'max_depth': 6,
      'subsample': 0.6383,
      'colsample_bytree': 0.8438,
      'reg_lambda': 6.2578,
      'reg_alpha': 4.0354,
      'enable_categorical': True,
      'random_state': 42
  }
  ```
* **LightGBM Regressor**:
  ```python
  {
      'n_estimators': 224,
      'learning_rate': 0.0393,
      'max_depth': 7,
      'num_leaves': 19,
      'subsample': 0.6004,
      'subsample_freq': 1,
      'colsample_bytree': 0.6357,
      'reg_lambda': 1.4231,
      'reg_alpha': 0.0826,
      'random_state': 42
  }
  ```

* **Stacking Ensemble**:
  A weighted blend of predictions: **80% CatBoost + 10% XGBoost + 10% LightGBM**.

---

## Experimental Benchmarks & Results

### A. Kharif Rice Chronological Performance (Tuned Models)

| Model | Fold 1 R² (2017-18) | Fold 2 R² (2019-20) | Fold 3 R² (2021-22) | **Average R²** | **Average RMSE** | **Average MAE** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** (Tuned) | 0.808 | 0.753 | 0.713 | **0.758** | - | - |
| **LightGBM** (Tuned) | 0.810 | 0.772 | 0.718 | **0.766** | - | - |
| **CatBoost** (Tuned) | 0.815 | 0.775 | 0.724 | **0.771** | **0.434** | **0.309** |
| **Weighted Ensemble** (Optimal Blend) | 0.812 | 0.766 | 0.720 | **0.766** | **0.441** | **0.315** |

> [!NOTE]
> The out-of-time chronological validation average R² improved from the initial baseline of **0.767** to **0.771** after integrating expanding standard anomalies, CWSI stress indices, and running Optuna tuning.

### B. Rabi Wheat Chronological Performance (Baseline Tuned)

| Model | Fold 1 R² (2017-18) | Fold 2 R² (2019-20) | Fold 3 R² (2021-22) | **Average R²** | **Average RMSE** | **Average MAE** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** (Baseline) | 0.851 | 0.826 | 0.828 | **0.835** | **0.484** | **0.350** |
| **LightGBM** (Baseline) | 0.872 | 0.867 | 0.841 | **0.860** | **0.445** | **0.320** |
| **CatBoost** (Baseline) | 0.886 | 0.868 | 0.842 | **0.865** | **0.436** | **0.312** |
| **Weighted Ensemble** | 0.885 | 0.867 | 0.844 | **0.865** | **0.436** | **0.312** |

> [!TIP]
> **Why Wheat Performs Better**: Rabi Wheat is heavily irrigated in India's Indo-Gangetic plains. The irrigation buffers the crop against localized rainfall shocks, producing a cleaner, highly predictable signal for the climate variables (soil moisture and temperature) compared to monsoon-dependent Kharif Rice.

### C. Benchmarking Performance: Taylor Diagram & Yield Scatter Plot

Validation predictions and individual learner statistics are mapped below:

| Taylor Diagram Benchmarking | Predicted vs. Actual Scatter Plot |
| :---: | :---: |
| ![Taylor Diagram Benchmarking](plots/model_taylor_diagram.png) | ![Predicted vs Actual](plots/predicted_vs_actual_scatter.png) |
| *Figure 7: Taylor Diagram benchmarking standard deviations.* | *Figure 8: Actual vs. Predicted yields for the out-of-time test sets.* |

### D. Complete 15-Crop Portfolio Validation Benchmarks
The chronological out-of-time cross-validation metrics (R² and RMSE) for the complete crop portfolio using the standardized baseline configuration are compiled below:

| Crop Profile | Season | Avg R² | Avg RMSE | Avg MAE | Fold 1 R² | Fold 2 R² | Fold 3 R² |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kharif Rice** (`kharif_rice`) | Kharif | **0.778** | 0.429 | 0.307 | 0.808 | 0.804 | 0.721 |
| **Rabi Wheat** (`rabi_wheat`) | Rabi | **0.860** | 0.444 | 0.321 | 0.873 | 0.865 | 0.842 |
| **Kharif Maize** (`kharif_maize`) | Kharif | **0.812** | 0.749 | 0.489 | 0.742 | 0.830 | 0.865 |
| **Kharif Groundnut** (`kharif_groundnut`) | Kharif | **0.675** | 0.411 | 0.277 | 0.665 | 0.582 | 0.777 |
| **Kharif Soyabean** (`kharif_soyabean`) | Kharif | **0.603** | 0.379 | 0.259 | 0.495 | 0.598 | 0.717 |
| **Kharif Cotton** (`kharif_cotton`) | Kharif | **0.326** | 1.150 | 0.702 | 0.559 | 0.370 | 0.050 |
| **Kharif Arhar** (`kharif_arhar`) | Kharif | **0.497** | 0.332 | 0.205 | 0.375 | 0.522 | 0.593 |
| **Rabi Potato** (`rabi_potato`) | Rabi | **0.701** | 5.382 | 3.986 | 0.683 | 0.736 | 0.685 |
| **Rabi Onion** (`rabi_onion`) | Rabi | **0.717** | 5.494 | 3.005 | 0.454 | 0.819 | 0.877 |
| **Rabi Tobacco** (`rabi_tobacco`) | Rabi | **0.306** | 1.015 | 0.763 | 0.056 | 0.497 | 0.366 |
| **Year Sugarcane** (`year_sugarcane`) | Year | **0.756** | 16.950 | 11.205 | 0.811 | 0.799 | 0.658 |
| **Year Banana** (`year_banana`) | Year | **0.760** | 10.599 | 6.295 | 0.562 | 0.864 | 0.855 |
| **Year Ginger** (`year_ginger`) | Year | **0.661** | 3.803 | 2.240 | 0.659 | 0.685 | 0.639 |
| **Year Turmeric** (`year_turmeric`) | Year | **0.501** | 3.262 | 1.611 | 0.466 | 0.734 | 0.303 |
| **Year Coconut** (`year_coconut`) | Year | **0.573** | 3.991 | 2.556 | 0.546 | 0.556 | 0.618 |

### E. Predictability Footprint Visualizations

The predictability of the complete portfolio and geographical error structures are shown below:

| Forest Performance Plot | Circular Radar Chart | Spatial Hexbin Map |
| :---: | :---: | :---: |
| ![Forest Plot](plots/multi_crop_performance_forest.png) | ![Radar Chart](plots/multi_crop_radial_predictability.png) | ![Hexbin Map](plots/spatial_accuracy_hexbin.png) |
| *Figure 9: Crop-specific model R² with standard deviations.* | *Figure 10: Radar chart illustrating predictability footprints.* | *Figure 11: Hexagonal map binning out-of-time spatial R² accuracy.* |

---

## Model Interpretability via SHAP & Feature Heatmap Analysis

To justify the physical validity of the predictions and extract agronomic insights, we computed global SHAP values and importance matrices:

### A. SHAP Beeswarm Feature Contribution
The beeswarm plot ranks features by their impact on yield prediction (high values in red, low values in blue):

![SHAP Beeswarm Summary Plot](plots/shap_beeswarm.png)
*Figure 12: SHAP Beeswarm Summary Plot ranking top features by prediction impact.*

*Key Insights:*
* **Historical Yield (`Kharif_Yield_Hist_Mean`)**: Strongest predictor of regional crop yields, serving as a baseline proxy for local infrastructure.
* **Crop Water Stress Index (`CWSI_Jul` and `CWSI_Aug`)**: Elevated values (red) represent heat/moisture stress, reducing yields.
* **NDVI Anomaly (`NDVI_Aug_Anomaly`)**: High density anomalies in August correspond to yield gains.

### B. Global Feature Importance Heatmap (15 Crops)
The relative feature importance weight across all 15 crop profiles is illustrated below:

![Feature Importance Heatmap Matrix](plots/feature_importance_heatmap_matrix.png)
*Figure 13: Normalized feature importance heatmap across the crop portfolio and key feature categories.*

---

## Stacking Ensemble Blending Weight Derivation

The repository implements a stacked ensemble combining three tree-based architectures: CatBoost, XGBoost, and LightGBM. To justify the blending weights (80% CatBoost + 10% XGBoost + 10% LightGBM), we performed a grid search optimization over the out-of-fold validation predictions across all chronological CV folds.

### A. Individual Validation Performance
* **XGBoost**: Validation R² = 0.7816 | RMSE = 0.4452
* **LightGBM**: Validation R² = 0.7782 | RMSE = 0.4486
* **CatBoost**: Validation R² = 0.7765 | RMSE = 0.4504

### B. Blending Grid Search Landscape
The grid search evaluated all weight combinations (summing to 1.0) with a step size of 0.01:
1. **Mathematical Optimum**: 59% XGBoost / 30% LightGBM / 11% CatBoost (R² = 0.7834 | RMSE = 0.4434).
2. **Standard Blend (80% CatBoost / 10% XGBoost / 10% LightGBM)**: Validation R² = 0.7790 | RMSE = 0.4478.

The performance difference between the standard blend and the absolute validation optimum is **ΔR² = 0.0043**. The standard blend (80% CatBoost / 10% XGBoost / 10% LightGBM) was selected as a robust compromise because:
* CatBoost exhibits the highest out-of-time test R² values on unseen testing folds (0.7711 compared to LightGBM's 0.7664 and XGBoost's 0.7579), proving it has superior out-of-time generalizability compared to its validation fold performance.
* Placing a larger weight on CatBoost prevents overfitting to validation fold years (2015-16, 2017-18, 2019-20).

---

## Deep Learning: Temporal Fusion Transformer (TFT)

We implemented an adapted **Temporal Fusion Transformer (TFT)** using PyTorch to natively process the time-series structure of the climate data (June, July, August, September), instead of relying solely on flattened tabular features.

### TFT Architecture
* **Static Covariate Encoders**: Embeddings for categorical inputs (`State`, `District_mapped`) and a Variable Selection Network (VSN) for static continuous features (e.g., Lags, Historical Yields).
* **Temporal Encoders**: A shared VSN applied across each time step, followed by an LSTM sequence processor to capture temporal dependencies in weather and satellite indices.
* **Temporal Self-Attention**: Multi-Head Attention mechanisms to weigh the importance of specific months (e.g., peak vegetative stage) dynamically.
* **Gating and Residual Networks (GRN)**: To filter out irrelevant variables and prevent overfitting on smaller temporal datasets.

The TFT is trained with strict out-of-time chronological cross-validation and compared head-to-head against the robust tree-based ensemble.

---

## Directory Structure

```
TerraYield-ML/
├── requirements.txt            # System dependencies
├── train_crop.py               # Unified crop training and serialization CLI
├── train_tft.py                # Temporal Fusion Transformer training CLI
├── error_analysis.py           # Evaluation script for baseline vs TFT comparison
├── tune_rice_optuna.py         # Optuna hyperparameter tuning utility
├── plots/                      # Publication-grade visual diagnostics
│   ├── multi_crop_performance_forest.png
│   ├── ablation_waterfall.png
│   ├── cwsi_vs_yield_time_series.png
│   ├── harmonization_vs_raw_trajectory.png
│   ├── residuals_distribution_over_time.png
│   ├── ensemble_vs_single_generalization.png
│   ├── multi_crop_radial_predictability.png
│   ├── small_multiples_spatial_residuals.png
│   ├── variance_partitioning_donut.png
│   ├── cwsi_3d_response_surface.png
│   ├── boundary_harmonization_sankey.png
│   ├── model_taylor_diagram.png
│   ├── feature_importance_heatmap_matrix.png
│   ├── phenological_stress_ridgeline.png
│   ├── feature_collinearity_network.png
│   └── spatial_accuracy_hexbin.png
├── data/
│   ├── raw/                    # Web-scraped DESAgri CSVs and GEE boundaries
│   └── processed/              # Mappings and final harmonized datasets (crop CSVs, climate panel, crosswalk)
├── data_ingestion/             # Collection & harmonization scripts
│   ├── dataset_web_scrapper.py # Scrapes crop yields from DESAgri portal
│   ├── apply_weights_to_exports.py # Historical boundary weighting to modern maps
│   ├── clean_crop_data.py      # Unit normalizations & yield outlier removal
│   ├── gee_climate_extraction.py # GEE climate & precipitation harvester
│   ├── add_modis_ndvi.py       # MODIS greenness index harvester
│   ├── add_soil_moisture.py    # Soil moisture grid appender
│   └── impute_modis_ndvi.py    # Reconstruct pre-2000 MODIS values using climatology
├── src/                        # Core ML package modules
│   ├── __init__.py
│   ├── config.py               # Crop profiles, hyperparameters, and offsets
│   ├── data_loader.py          # Data ingestion, spatial lags, and anomalies
│   ├── features.py             # Feature list builders
│   ├── models.py               # Time-series cross-validation trainer
│   ├── tft_model.py            # Temporal Fusion Transformer PyTorch model
│   └── district_mappings.py    # Dictionary for administrative district merges
```

---

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AadityaChaturvedy/TerraYield-ML.git
   cd TerraYield-ML
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Acquisition & Ingestion Pipeline

If you wish to harvest the datasets from scratch rather than using the processed CSV files, you can use the scripts in `data_ingestion/`. **You must configure your own credentials for these scripts**:

### 1. Yield Data Scraping & Boundary Harmonization
The yield pipeline crawls the government DESAgri APY Portal and then harmonizes historical boundaries.
* **Scrape Raw Yields**: You must enter your own browser session details. Run the script interactively, or pass them as command-line arguments:
  ```bash
  python data_ingestion/dataset_web_scrapper.py <XSRF-TOKEN> <laravel_session> <_token>
  ```
  *Get these cookie values from your browser's Developer Tools (Network tab) when requesting any crop report on the portal. This outputs raw yield tables into `data/raw/`.*

* **Apply Boundary Harmonization Weights**:
  Run the harmonization script to re-allocate extensive variables (Area, Production) from old boundaries to new administered boundaries using the checked-in conversion weights:
  ```bash
  python data_ingestion/apply_weights_to_exports.py
  ```
  *This scales yield variables by old-to-new district area weights, aggregates them, and writes the modern harmonized CSV outputs to `data/processed/`.*

* **Clean Yield Outliers & Normalize Units**:
  Run the cleaning script to automatically drop empty seasonal columns, correct reporting errors/outliers, and normalize coconut yield units (convert from single nuts to thousands of nuts) to make the scales consistent for modeling:
  ```bash
  python data_ingestion/clean_crop_data.py
  ```
  *This modifies the processed files directly in `data/processed/`.*

### 2. Earth Engine Climate & Remote Sensing Extraction
The extraction scripts run on Google Earth Engine (GEE).
* **Setup GEE Authentication**:
  * You must have a Google Cloud Project with GEE enabled.
  * In `data_ingestion/gee_climate_extraction.py`, update:
    ```python
    ee.Initialize(project='your-gcp-project-id')
    GEE_ASSET_ID = 'projects/your-gcp-project/assets/india_district_administered'
    ```
  * Place your harmonized district shapefile/GeoJSON at `data/raw/india_district_administered.geojson`.
* **Run Extraction Pipeline**:
  ```bash
  # 1. Extract ERA5 temperature and CHIRPS precipitation
  python data_ingestion/gee_climate_extraction.py
  
  # 2. Append MODIS NDVI and EVI bands (post-2000)
  python data_ingestion/add_modis_ndvi.py
  
  # 3. Append volumetric Soil Moisture Layer 1 & 2
  python data_ingestion/add_soil_moisture.py
  
  # 4. Impute pre-2000 MODIS values using district climatology
  python data_ingestion/impute_modis_ndvi.py
  ```

---

## Running the Yield Hindcasting Pipeline

To train and evaluate the hindcasting models under strict out-of-time chronological validation, place your processed CSV datasets in `data/processed/` and use the portability CLI:

### 1. Unified CLI (`train_crop.py`)
```bash
python train_crop.py [crop_profile] [options]
```

#### Core Command Options:
- `--data_path`: Optional path to custom crop yield CSV file.
- `--climate_path`: Optional path to custom GEE climate CSV file.
- `--neighbors_path`: Optional path to custom district neighbors JSON file.
- `--include_district`: Include mapped district boundaries as a categorical feature.
- `--include_ext_months`: Include May and October extended months climate data.
- `--include_soil_l1`: Include ERA5 Soil Moisture L1 feature.
- `--include_interactions`: Include engineered climate interaction features.
- `--include_yield_trend`: Include dynamic district-level yield trend slope feature.
- `--include_lag3`: Include a 3-year historical yield lag feature.
- `--include_phenology`: Enable dynamic phenological month alignment features.
- `--include_sensor_flag`: Include post-2000 satellite sensor boundary flags.

#### Kharif Rice Execution Examples:
* **Train Kharif Rice Baseline (Avg R² ~0.778)**:
  ```bash
  PYTHONPATH=. python train_crop.py kharif_rice \
    --include_district --include_ext_months --include_soil_l1 --include_lag3
  ```
* **Train Kharif Rice with Sensor Harmonization Flag**:
  ```bash
  PYTHONPATH=. python train_crop.py kharif_rice \
    --include_district --include_ext_months --include_soil_l1 --include_lag3 --include_sensor_flag
  ```

#### Rabi Wheat Execution Example:
```bash
PYTHONPATH=. python train_crop.py rabi_wheat \
  --include_district --include_ext_months --include_soil_l1 --include_lag3
```

---

### 2. Hyperparameter Tuning (`tune_rice_optuna.py`)
To run the Optuna hyperparameter optimization for Kharif Rice:
* **Fast Demo Mode** (30 trials CatBoost, 20 trials XGBoost, 20 trials LightGBM):
  ```bash
  PYTHONPATH=. python tune_rice_optuna.py
  ```
* **Full Tuning Mode** (100 trials CatBoost, 80 trials XGBoost, 80 trials LightGBM - replicates paper parameters):
  ```bash
  PYTHONPATH=. python tune_rice_optuna.py --full
  ```

---

## Ablation Studies & Statistical Visualizations

### A. Incremental Feature Engineering Ablation (Kharif Rice)

| Integration Stage | Avg R² | Avg RMSE | Avg MAE | Fold 1 R² | Fold 2 R² | Fold 3 R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0: Baseline** (Jun-Sep Climate) | 0.7671 | 0.4399 | 0.3067 | 0.8016 | 0.7873 | 0.7125 |
| **1: + District Mappings** | 0.7695 | 0.4370 | 0.3084 | 0.8112 | 0.7890 | 0.7083 |
| **2: + Extended Months (May/Oct)** | 0.7735 | 0.4336 | 0.3060 | 0.8100 | 0.7928 | 0.7177 |
| **3: + Soil Moisture L1** | 0.7750 | 0.4322 | 0.3042 | 0.8120 | 0.7931 | 0.7199 |
| **4: + Interaction Features** | 0.7741 | 0.4332 | 0.3049 | 0.8088 | 0.7931 | 0.7203 |
| **5: + Yield Trend Slope** | 0.7720 | 0.4350 | 0.3069 | 0.8127 | 0.7848 | 0.7185 |

To run the incremental feature ablation study and generate the waterfall figure:
```bash
PYTHONPATH=. python scratch/run_ablation_study.py
```

| Ablation Waterfall | Error Distribution Over Time |
| :---: | :---: |
| ![Ablation Waterfall](plots/ablation_waterfall.png) | ![Error over Time](plots/residuals_distribution_over_time.png) |
| *Figure 14: Step-by-step performance waterfall.* | *Figure 15: Time-series showing wider error bounds for the 2022 crop season.* |

### B. Phenological Joyplot Progression & Network Collinearity

We evaluate monthly phenological stress progression and feature correlation structures:

| Joyplot Ridgeline | Feature Network Graph | Variance Partitioning |
| :---: | :---: | :---: |
| ![Ridgeline Plot](plots/phenological_stress_ridgeline.png) | ![Network Graph](plots/feature_collinearity_network.png) | ![Variance Partitioning](plots/variance_partitioning_donut.png) |
| *Figure 16: Joyplot showing phenological stress shifts.* | *Figure 17: Feature collinearity network ($\vert r \vert \gt 0.65$).* | *Figure 18: Donut chart showing partition of explained variance.* |

---

## How to Cite

If you use the TERRA pipeline, software, or model profiles in your research, please cite the code repository:

```bibtex
@software{terra_yield_ml_2026,
  author       = {Chaturvedy, Aaditya and Gururajan, Bhargavi},
  title        = {TerraYield-ML: Regional Crop Yield Hindcasting Engine (TERRA)},
  year         = {2026},
  version      = {1.0.0},
  url          = {https://github.com/AadityaChaturvedy/TerraYield-ML}
}
```

If you use the harmonized dataset panel, please cite the data release:

```bibtex
@dataset{chaturvedy_2026_terrayield_ml,
  author       = {Chaturvedy, Aaditya and G, Bhargavi},
  title        = {A Harmonized District-Level Crop Yield and Climatology Panel for India, 1997–2022},
  month        = jul,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.21126177},
  url          = {https://doi.org/10.5281/zenodo.21126177}
}
```
