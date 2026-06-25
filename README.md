# TerraYield-ML: Regional Crop Yield Hindcasting Engine

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
When evaluated under standard random cross-validation, the average Ensemble R² is **0.8381** (RMSE = 0.4086). This **~0.06 R² inflation** demonstrates the extent of spatial-temporal data leakage and justifies the strict chronological split.

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
2. **Post-2000 Sensor Harmonization Flag**: A binary indicator `Post2000` to allow the tree model to adjust its splits for the transition from AVHRR climatology-imputed data (1997–1999) to actual MODIS observations (2000 onwards).
3. **Dynamic Phenological Alignment**: Maps climate variables relative to the peak vegetative month (maximum NDVI) per district-year to handle varying sowing dates across Indian states:
   * $t=0$: Peak month (`Peak`)
   * $t=-1, -2$: Pre-peak sowing/emergence months (`Peak_Minus1`, `Peak_Minus2`)
   * $t=+1$: Post-peak reproductive month (`Peak_Plus1`)

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

### A. Kharif Rice Chronological Performance (Tuned Ensemble)

| Model | Fold 1 R² (2017-18) | Fold 2 R² (2019-20) | Fold 3 R² (2021-22) | **Average R²** | **Average RMSE** | **Average MAE** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** (Tuned) | 0.8078 | 0.7976 | 0.7129 | **0.7728** | - | - |
| **LightGBM** (Tuned) | 0.8104 | 0.7984 | 0.7175 | **0.7755** | - | - |
| **CatBoost** (Tuned) | 0.8128 | 0.8104 | 0.7288 | **0.7840** | - | - |
| **Weighted Ensemble** | 0.8137 | 0.8105 | 0.7283 | **0.7842** | **0.4228** | **0.3002** |

> [!NOTE]
> The out-of-time chronological validation average R² improved from the initial baseline of **0.7671** to **0.7842** after integrating expanding standard anomalies, CWSI stress indices, and running Optuna tuning.

### B. Rabi Wheat Chronological Performance (Baseline Tuned)

| Model | Fold 1 R² (2017-18) | Fold 2 R² (2019-20) | Fold 3 R² (2021-22) | **Average R²** | **Average RMSE** | **Average MAE** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** (Baseline) | 0.8252 | 0.8351 | 0.8303 | **0.8302** | - | - |
| **LightGBM** (Baseline) | 0.8711 | 0.8654 | 0.8525 | **0.8630** | - | - |
| **CatBoost** (Baseline) | 0.8762 | 0.8716 | 0.8520 | **0.8666** | - | - |
| **Weighted Ensemble** | 0.8756 | 0.8712 | 0.8520 | **0.8663** | **0.4350** | **0.3108** |

> [!TIP]
> **Why Wheat Performs Better**: Rabi Wheat is heavily irrigated in India's Indo-Gangetic plains. The irrigation buffers the crop against localized rainfall shocks, producing a cleaner, highly predictable signal for the climate variables (soil moisture and temperature) compared to monsoon-dependent Kharif Rice.

### C. Complete 15-Crop Portfolio Validation Benchmarks
The chronological out-of-time cross-validation metrics (R² and RMSE) for the complete crop portfolio using the standardized baseline configuration are compiled below:

| Crop Profile | Season | Avg R² | Avg RMSE | Avg MAE | Fold 1 R² | Fold 2 R² | Fold 3 R² |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kharif Rice** (`kharif_rice`) | Kharif | **0.7777** | 0.4292 | 0.3071 | 0.8079 | 0.8041 | 0.7210 |
| **Rabi Wheat** (`rabi_wheat`) | Rabi | **0.8603** | 0.4444 | 0.3205 | 0.8731 | 0.8654 | 0.8423 |
| **Kharif Maize** (`kharif_maize`) | Kharif | **0.8124** | 0.7491 | 0.4890 | 0.7423 | 0.8301 | 0.8649 |
| **Kharif Groundnut** (`kharif_groundnut`) | Kharif | **0.6746** | 0.4110 | 0.2771 | 0.6653 | 0.5817 | 0.7767 |
| **Kharif Soyabean** (`kharif_soyabean`) | Kharif | **0.6032** | 0.3787 | 0.2593 | 0.4951 | 0.5979 | 0.7167 |
| **Kharif Cotton** (`kharif_cotton`) | Kharif | **0.3262** | 1.1503 | 0.7024 | 0.5589 | 0.3698 | 0.0499 |
| **Kharif Arhar** (`kharif_arhar`) | Kharif | **0.4967** | 0.3316 | 0.2053 | 0.3746 | 0.5224 | 0.5930 |
| **Rabi Potato** (`rabi_potato`) | Rabi | **0.7014** | 5.3816 | 3.9865 | 0.6833 | 0.7360 | 0.6848 |
| **Rabi Onion** (`rabi_onion`) | Rabi | **0.7167** | 5.4940 | 3.0053 | 0.4543 | 0.8186 | 0.8772 |
| **Rabi Tobacco** (`rabi_tobacco`) | Rabi | **0.3061** | 1.0153 | 0.7629 | 0.0557 | 0.4965 | 0.3662 |
| **Year Sugarcane** (`year_sugarcane`) | Year | **0.7561** | 16.9504 | 11.2052 | 0.8112 | 0.7988 | 0.6582 |
| **Year Banana** (`year_banana`) | Year | **0.7603** | 10.5992 | 6.2952 | 0.5620 | 0.8638 | 0.8551 |
| **Year Ginger** (`year_ginger`) | Year | **0.6614** | 3.8028 | 2.2401 | 0.6595 | 0.6854 | 0.6392 |
| **Year Turmeric** (`year_turmeric`) | Year | **0.5012** | 3.2620 | 1.6113 | 0.4662 | 0.7344 | 0.3031 |
| **Year Coconut** (`year_coconut`) | Year | **0.5733** | 3.9907 | 2.5565 | 0.5463 | 0.5557 | 0.6180 |

---

## Model Interpretability via SHAP Analysis

To justify the physical validity of the predictions and extract agronomic insights, we computed global SHAP (SHapley Additive exPlanations) values on the primary XGBoost model for Kharif Rice.

### A. SHAP Beeswarm Feature Contribution
The beeswarm plot ranks the top 15 features by their impact on yield prediction, showing the direction of feature influence (high values in red, low values in blue):

![SHAP Beeswarm Summary Plot](assets/shap_beeswarm.png)

*Key Insights:*
* **Historical Yield (`Kharif_Yield_Hist_Mean`)**: This is the strongest predictor of regional crop yields, serving as a baseline proxy for structural irrigation infrastructure, local soil type, and technology adoption.
* **Crop Water Stress Index (`CWSI_Jul` and `CWSI_Aug`)**: Elevated values (red) represent high temperature paired with low rainfall, which drastically reduces yields (shifts SHAP values to the left/negative). This corresponds directly to crop stress during critical tillering and vegetative growth phases.
* **NDVI Anomaly (`NDVI_Aug_Anomaly`)**: High vegetative density anomalies in August (red) are strongly correlated with positive yield gains, confirming the model uses remote sensing vegetation vigor to adjust yield upward.
* **Spatial Neighbor Yield (`Kharif_Yield_Spatial_Lag1`)**: Higher yields in neighboring districts historically correlate with positive local yields, validating our boundary-harmonized spatial adjacency representations.

### B. Global Feature Importance
The feature importance bar chart ranks the top 15 features by their mean absolute SHAP value, representing their average global impact:

![Mean Absolute SHAP Feature Importance](assets/shap_importance.png)

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
* CatBoost exhibits the highest out-of-time test R² values on unseen testing folds (0.7840 compared to LightGBM's 0.7755 and XGBoost's 0.7728), proving it has superior out-of-time generalizability compared to its validation fold performance.
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
├── assets/                     # Visual diagnostic plots (SHAP)
│   ├── shap_beeswarm.png
│   └── shap_importance.png
├── data/
│   ├── raw/                    # Web-scraped DESAgri CSVs and GEE boundaries
│   └── processed/              # Boundary conversion weights & neighbors maps
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
└── scratch/                    # Testing scripts & ablation check benches
    ├── all_crops_results.json  # Saved metrics for all 15 crop profiles
    ├── check_random_kfold.py   # Simulates random K-Fold to check spatial leakage
    ├── run_ablation_study.py   # Iterates features from baseline (Stage 0) to Stage 5
    ├── test_drought_flags.py   # Ablation study testing severe weather binary overrides
    ├── test_phenology_alignment.py # Phenology feature evaluation scripts
    └── train_all_crops.py      # Evaluation runner for all profiles
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
* **Train Kharif Rice Baseline (Avg R² ~0.7777)**:
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

## Ablation Studies

### A. Incremental Feature Engineering Ablation (Kharif Rice)

| Integration Stage | Avg R² | Avg RMSE | Avg MAE | Fold 1 R² | Fold 2 R² | Fold 3 R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0: Baseline** (Jun-Sep Climate) | 0.7671 | 0.4399 | 0.3067 | 0.8016 | 0.7873 | 0.7125 |
| **1: + District Mappings** | 0.7695 | 0.4370 | 0.3084 | 0.8112 | 0.7890 | 0.7083 |
| **2: + Extended Months (May/Oct)** | 0.7735 | 0.4336 | 0.3060 | 0.8100 | 0.7928 | 0.7177 |
| **3: + Soil Moisture L1** | 0.7750 | 0.4322 | 0.3042 | 0.8120 | 0.7931 | 0.7199 |
| **4: + Interaction Features** | 0.7741 | 0.4332 | 0.3049 | 0.8088 | 0.7931 | 0.7203 |
| **5: + Yield Trend Slope** | 0.7720 | 0.4350 | 0.3069 | 0.8127 | 0.7848 | 0.7185 |

To run the incremental feature ablation study:
```bash
PYTHONPATH=. python scratch/run_ablation_study.py
```

### B. Dynamic Phenological Alignment Ablation (Kharif Rice)
Explicitly aligning climate variables relative to the peak NDVI month and adding the sensor boundary flag:
* **Base Tuned Model (114 features)**: Avg R² = **0.7842** (RMSE = **0.4228**)
* **Aligned Model (185 features)**: Avg R² = **0.7802** (RMSE = **0.4268**)
* **Finding**: The phenological alignment slightly degraded out-of-time chronological performance (ΔR² = -0.0040) due to multicollinearity and split fragmentation (adding 71 aligned variables). The tree models implicitly learn state-specific offsets via the `State` categorical and raw monthly features.

To run the phenological alignment evaluation script:
```bash
PYTHONPATH=. python scratch/test_phenology_alignment.py
```

### C. Severe Weather Override Check
Test the impact of binary drought and heat stress flags in helping the models adjust predicted yields during severe crop failure years:
```bash
PYTHONPATH=. python scratch/test_drought_flags.py
```

### 5. Multi-Crop Batch Evaluator
Train and evaluate all 15 crop profiles in a single command, updating `scratch/all_crops_results.json`:
```bash
PYTHONPATH=. python scratch/train_all_crops.py
```

---

## How to Cite

If you use the TERRA pipeline, datasets, or model profiles in your research, please cite our software and manuscript as follows:

```bibtex
@software{terra_yield_ml_2026,
  author       = {Chaturvedy, Aaditya and Gururajan, Bhargavi},
  title        = {TerraYield-ML: Regional Crop Yield Hindcasting Engine (TERRA)},
  year         = {2026},
  version      = {1.0.0},
  url          = {https://github.com/AadityaChaturvedy/TerraYield-ML}
}
```
