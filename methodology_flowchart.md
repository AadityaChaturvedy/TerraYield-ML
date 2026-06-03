# TERRA: Regional Crop Yield Hindcasting Engine Methodology

This document details the step-by-step methodology and architectural flow of the **TERRA (Regional Yield Hindcasting Engine)**. 

The pipeline is designed to ingest multi-source agronomic statistics and climate datasets, clean and harmonize administrative boundaries, engineer leakage-free spatial-temporal features, and execute a robust stacking ensemble evaluated under strict chronological cross-validation.

---

## Technical Methodology Flowchart

```mermaid
flowchart TD
    %% Define styles and classes
    classDef ingest fill:#f9f0ff,stroke:#d3adf7,stroke-width:2px,color:#000;
    classDef clean fill:#e6f7ff,stroke:#91d5ff,stroke-width:2px,color:#000;
    classDef feature fill:#f6ffed,stroke:#b7eb8f,stroke-width:2px,color:#000;
    classDef model fill:#fff2e8,stroke:#ffbb96,stroke-width:2px,color:#000;
    classDef val fill:#fff0f6,stroke:#ffadd2,stroke-width:2px,color:#000;

    subgraph INGESTION["Phase 1: Ingestion & Boundary Standardization"]
        A1["DESAgri Yield API Scraper"]:::ingest
        A2["ERA5-Land & MODIS GEE Assets"]:::ingest
        A3["Modern Administered Shapefiles"]:::ingest
        
        A1 & A3 --> B1["Apply Old-to-New Boundary Weights"]:::ingest
        B1 --> B2["Modern Harmonized Tabular Yields"]:::ingest
    end

    subgraph CLEANING["Phase 2: Outlier Cleaning & Target Normalization"]
        B2 --> C1["Crop-Specific Max Yield Capping"]:::clean
        C1 --> C2["Normalizing Coconut Units (/1000 to Thousand Nuts)"]:::clean
        C2 --> C3["Drop -1.0 Missing Columns & Rows"]:::clean
    end

    subgraph CLIMATE["Phase 3: Climatology & Satellite Processing"]
        A2 & A3 --> D1["Spatial-Temporal Raster Aggregation"]:::clean
        D1 --> D2["MODIS NDVI/EVI Imputation (Pre-2000 Climatology)"]:::clean
        D2 --> D3["District Monthly Climate CSVs (1997-2022)"]:::clean
    end

    subgraph FEATURES["Phase 4: Feature Engineering (No-Leakage)"]
        C3 & D3 --> E1["Merge Crop Statistics & Climate Datasets"]:::feature
        
        E1 --> F1["Temporal Lags (Lag-1, Lag-2, Lag-3 Yields)"]:::feature
        E1 --> F2["Historical Expanding Means (Yield Trend Slope)"]:::feature
        E1 --> F3["Spatial Neighbor Lags (Weighted Adjacency Yields)"]:::feature
        E1 --> F4["Expanding Z-Score Climatic Anomalies"]:::feature
        E1 --> F5["Crop Water Stress Index (CWSI = Temp_Z - Precip_Z - Soil_Z)"]:::feature
    end

    subgraph MODELING["Phase 5: Stacking Ensemble & Strict Evaluation"]
        F1 & F2 & F3 & F4 & F5 --> G1["State-Specific Quantile Target Clipping"]:::model
        
        G1 --> H1["XGBoost Regressor (10%)"]:::model
        G1 --> H2["LightGBM Regressor (10%)"]:::model
        G1 --> H3["CatBoost Regressor (80%)"]:::model
        
        H1 & H2 & H3 --> I1["Robust Blend Stacking (0.10/0.10/0.80)"]:::model
        
        I1 --> J1["Out-of-Time Chronological Validation"]:::val
        
        subgraph CV_SPLIT["Validation Split Scheme"]
            J1 --> K1["Train: Years < Y_n-4"]:::val
            J1 --> K2["Validate: [Y_n-3, Y_n-2] (Early Stop)"]:::val
            J1 --> K3["Test: [Y_n-1, Y_n] (Out-of-Time Folds)"]:::val
        end
        
        K3 --> L1["Final Yield Hindcast Outputs"]:::val
    end
```

---

## Architectural Breakdown

### 1. Ingestion & Boundary Standardization
* **Yield API Scraper**: Crolls the government DESAgri APY Portal to collect historical crop production, area, and yield metrics (1997-2022).
* **Boundary Weights Harmonization**: Corrects for administrative changes (e.g., district splits or reorganizations). Yield statistics from older boundaries are reallocated using verified spatial boundary weights to ensure consistency.

### 2. Cleaning & Normalization
* **Outlier Capping**: Replaces extreme yield outliers (representing administrative data entry errors) with `-1.0` and drops them before feature mapping.
* **Unit Normalization**: Coconut yield and production are normalized (divided by 1000) to scale the metrics into thousands of nuts, preventing extreme scale differences from skewing multi-crop models.
* **Column Pruning**: Detects and drops empty seasonal columns that contain exclusively missing markers.

### 3. Climate & Satellite Processing
* **Raster Aggregation**: Processes daily climate datasets (ERA5-Land temperature, precipitation, soil moisture) and satellite imagery (MODIS NDVI/EVI) to produce district-level monthly averages.
* **MODIS Imputation**: For years before 2000 (pre-MODIS era), satellite NDVI/EVI indexes are imputed using district-specific long-term historical climatology.

### 4. Leakage-Free Feature Engineering
To prevent spatial-temporal data leakage (which inflates standard cross-validation results by ~0.06 $R^2$), all engineered features are calculated sequentially in time:
* **Temporal Lags**: Shifted yield, area, and production records (1, 2, and 3-year lags).
* **Expanding Means**: District-level historical averages computed using only the data prior to the target year.
* **Climatic Z-Scores**: Anomaly variables computed as:
  $$Z_{d, y, m} = \frac{x_{d, y, m} - \mu_{d, <y, m}}{\sigma_{d, <y, m} + \epsilon}$$
* **Crop Water Stress Index (CWSI)**: A thermal-water stress indicator calculated by combining temperature, precipitation, and soil moisture Z-scores.
* **Spatial Neighbor Lags**: Average yields of adjacent neighboring districts from the preceding year.

### 5. Stacking Ensemble & Strict Evaluation
* **Ensemble Regressor**: Uses a stacked combination of **CatBoost** (80%), **XGBoost** (10%), and **LightGBM** (10%).
* **Chronological Split**: Enforces an Out-of-Time Chronological Validation Scheme (Train: $Y_{<n-4}$, Validate: $[Y_{n-3}, Y_{n-2}]$ for early stopping, Test: $[Y_{n-1}, Y_n]$) to test model performance on unseen future years.
