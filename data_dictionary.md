# TerraYield-ML Data Dictionary

This document details the schema, encoding schemes, source datasets, and preprocessing rules for the TerraYield-ML datasets.

---

## 1. Crop Yield CSV Datasets
Filename pattern: `data/crop_*_season_year_wide_harmonized.csv`

### Column Definitions
| Column Name | Unit | Source | Description |
| :--- | :--- | :--- | :--- |
| **Crop_Name** | string | DESAgri | Name of the crop (e.g., rice, wheat, sugarcane). |
| **State** | string | DESAgri | Harmonized State name (capitalized). |
| **District** | string | DESAgri / Crosswalk | Harmonized District name matching the climate boundary keys. |
| **Year** | integer | DESAgri | Calendar Year of harvest (1997 to 2022). |
| **{Season}_Area** | hectares (ha) | DESAgri | Area sown under {Season} (Kharif, Rabi, Autumn, Winter, Summer, Year). |
| **{Season}_Production** | tonnes (t) * | DESAgri | Crop production under {Season}. |
| **{Season}_Yield** | tonnes/ha (t/ha) * | Derived | Crop yield under {Season}, computed as: `Production / Area`. |

*\* Note on Coconut: Coconut production is measured in **thousands of nuts**, and coconut yield is in **thousand nuts/hectare** (thousands of nuts/ha). All other crops use metric tonnes (t) and tonnes/hectare (t/ha).

*\* Data Licensing Note: All crop yield, area, and production data are sourced from the Directorate of Economics and Statistics (DESAgri), Government of India, and are re-released in this panel under the terms of the Government Open Data License – India (GODL-India).*

### Outlier Clipping Thresholds
To remove extreme reporting anomalies (e.g., typos in raw government datasets), yields are clipped at the 99th percentile or the crop-specific maximum threshold (defined in `data_loader.py` and `clean_crop_data.py`):
- **Wheat / Rice / Groundnut / Tobacco**: 10.0 t/ha
- **Maize / Cotton Lint**: 15.0 t/ha
- **Turmeric**: 30.0 t/ha
- **Ginger**: 50.0 t/ha
- **Soybean / Arhar (Pigeon Pea)**: 8.0 t/ha
- **Onion / Potato / Coconut**: 100.0 t/ha (or thousand nuts/ha for coconut)
- **Banana**: 120.0 t/ha
- **Sugarcane**: 250.0 t/ha

---

## 2. Standalone Mappings Crosswalk
Filename: `data/district_name_mappings.csv`

Documents the complete alignment between administrative boundaries from the agricultural census (DESAgri) and the climate polygon grid (shapefile).
- **raw_name**: The original district string in the yield crop files.
- **harmonized_name**: The canonical district key used in the climate panel dataset.
- **state**: The state enclosing the district.
- **mapping_type**: Classification of change: `spelling variant`, `rename`, or `split`.
- **parent_district**: The parent/historical district enclosing the raw boundary (for splits).
- **notes**: Background context on the bifurcations or spelling corrections.

---

## 3. District Climate Panel Dataset
Filename: `data/district_climate_data_1997_2022.csv`

A wide-format panel dataset containing monthly-aggregated climate and remote sensing variables for 666 districts spanning 26 years (1997–2022).

### Identifier Columns

Unlike the crop yield CSVs, the climate panel uses a different naming 
convention for its identifier columns, inherited from the underlying GEE 
boundary shapefile:

- **NAME_1**: State name — equivalent to `State` in the crop yield files.
- **NAME_2**: Harmonized district name — equivalent to `District` in the crop 
  yield files. Values are consistent with the `harmonized_name` column in 
  `district_name_mappings.csv`.

Users merging this file with the crop yield CSVs should join on 
`State == NAME_1` and `District == NAME_2` (see `demo_load.py` for a 
reference implementation).

### Variables & Source Datasets
- **Precipitation**: Sum of monthly precipitation (mm) aggregated from **CHIRPS Daily (v2.0)** (Resolution: ~5km).
- **Temperature**: Mean monthly 2m air temperature (°C) converted from Kelvin, aggregated from **ECMWF ERA5-Land Monthly Aggregated** (Resolution: ~9km).
- **Soil Moisture Layer 1**: Mean monthly volumetric soil water (m³/m³) for layer 1 (0–7cm), from **ECMWF ERA5-Land Monthly Aggregated**.
- **Soil Moisture Layer 2**: Mean monthly volumetric soil water (m³/m³) for layer 2 (7–28cm), from **ECMWF ERA5-Land Monthly Aggregated**.
- **NDVI / EVI**: Vegetation indices (NDVI/EVI) from **MODIS MOD13A1 (v061)** (Resolution: 500m), scaled by `0.0001` (values range from -0.2 to 1.0). Computed for the June–September calendar window only, across all districts regardless of locally grown crop (not limited to Kharif-cropped areas).

### Cropland Masking
All climate and remote sensing variables aggregated within GEE are spatially masked using the **MODIS MCD12Q1 (v061)** land cover dataset. Only pixels classified as Croplands (IGBP Class 12) or Cropland/Natural Vegetation Mosaics (IGBP Class 14) in the corresponding year are included in the district-level spatial mean computation.

### GEE Month-Indexing Convention
Climate variables in the wide-format table are prefixed with a monthly sequence number running from `0` through `311` (representing 312 total months between January 1997 and December 2022).
- **Index `m`** corresponds to:
  - **Year**: `1997 + m // 12`
  - **Month**: `m % 12 + 1` (where 1 = January, 12 = December)
- **Index `0`**: January 1997 (`0_Precip_Sum`, `0_Temp_Mean`, `0_Soil_L1`, `0_Soil_L2`)
- **Index `1`**: February 1997
- **Index `12`**: January 1998
- **Index `311`**: December 2022 (`311_Precip_Sum`, `311_Temp_Mean`, `311_Soil_L1`, `311_Soil_L2`)

#### Vegetation Index (NDVI/EVI) Imputation
MODIS vegetation indices are computed for the June–September calendar window only, across all districts regardless of locally grown crop (not limited to Kharif-cropped areas).
- **Months offset**: June (`5`), July (`6`), August (`7`), September (`8`) of each year.
- **Pre-MODIS Era (1997–1999)**: Backfilled using the district-level climatological average for that specific month computed over the MODIS active years (2000–2022).

*\* Climate Licensing Note: ERA5-Land variables contain modified Copernicus Climate Change Service information [2026] and are licensed under CC-BY-4.0. CHIRPS (UCSB) and MODIS (NASA) datasets are in the public domain.
