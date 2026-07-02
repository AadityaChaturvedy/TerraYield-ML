## Title

**A Harmonized District-Level Crop Yield and Climatology Panel for India, 1997–2022**

---

## Abstract

District-level crop yield estimation in India is constrained by two persistent data challenges: administrative boundary instability, with over 120 new districts created between 1997 and 2022, and the misalignment of climate and satellite predictors with these shifting administrative units. We present a harmonized, analysis-ready panel dataset that resolves both issues, combining Directorate of Economics and Statistics (DESAgri) crop yield records for 15 major crops with monthly ERA5-Land reanalysis, CHIRPS precipitation, and MODIS vegetation indices, aggregated to a consistent set of 666 Indian districts over a 26-year period. Historical district splits and renames were reconciled to stable administrative boundaries using an area-weighted cropland allocation procedure, and all climate variables were cropland-masked using MODIS land-cover classification to isolate agriculturally relevant signals. The dataset is released with a fully documented boundary crosswalk, a data dictionary, and validation evidence confirming key-matching integrity and baseline predictive usability, intended to support reproducible research in agricultural remote sensing, climate risk modeling, and crop yield hindcasting/nowcasting for India.

---

## 1. Background & Summary

Accurate and localized crop yield data is essential for food security monitoring, agricultural policy planning, and weather-index insurance design. In India, district-level crop statistics are compiled annually by state governments under the Directorate of Economics and Statistics (DESAgri) and published via the Area, Production, and Yield (APY) Portal. While these records span several decades, their utility for machine learning and climate-risk research is limited by two major bottlenecks:

1. **Administrative boundary changes.** Over 120 new districts were carved out of older parent districts between 1997 and 2022, creating sharp artificial discontinuities in area and production time series that are frequently mishandled or ignored in downstream analyses.
2. **Missing spatial-temporal alignment.** Environmental predictors — reanalysis weather and satellite vegetation indices — are rarely aggregated to match the shifting administrative boundaries of the underlying crop reporting units, producing spatial mismatches that degrade both descriptive and predictive analysis.

The TerraYield-ML dataset resolves these challenges by providing a harmonized, district-level spatio-temporal crop yield panel matched with monthly-aggregated climatological variables for 666 districts in India across 26 years (1997–2022). Historical district splits were reconciled to stable parent boundaries using area-weighted cropland pixel allocation. Climatological variables were spatially aggregated using Google Earth Engine (GEE) and masked with MODIS cropland layers to filter out non-agricultural signal. The result is a clean, ready-to-use, publicly documented dataset designed to support agricultural hindcasting and nowcasting research, and more broadly any study requiring a stable long-run administrative panel for India.

This dataset was developed as part of the TerraYield modeling framework (Chaturvedy & Gururajan, in preparation), which uses this panel to benchmark district-level yield prediction under anti-leakage chronological validation. This Data Descriptor documents the underlying panel independently and in full, including the complete boundary harmonization crosswalk and validation evidence not previously disclosed in that companion work, to support reuse by the broader research community beyond the original modeling application.

---

## 2. Methods

The data integration pipeline consists of three components: agricultural yield collection, boundary harmonization, and cropland-masked climate extraction.

```
                              [ DESAgri APY Portal ]
                                         │
                                         ▼
                            [ Raw District Yields ]
                                         │
                                         ▼
                      [ District Boundary Harmonization ]
                     (Reallocated to Stable Parent Boundaries)
                                         │
                                         ▼
                         [ Harmonized Yield Data (15 Crops) ]
                                         │
                                         ▼
                       [ Cropland-Masked GEE Aggregation ]
                         (ERA5-Land, CHIRPS, MODIS)
                                         │
                                         ▼
                            [ Inner Join on Keys ]
                                         │
                                         ▼
                            [ TerraYield-ML Panel ]
```

### 2.1. Agricultural Yield Data and Boundary Harmonization

Crop area, production, and yield records were collected from the DESAgri APY portal for 15 major crops: rice, wheat, maize, groundnut, soyabean, cotton (lint), arhar (pigeon pea/tur), potato, onion, tobacco, sugarcane, banana, ginger, turmeric, and coconut.

Because new districts were continuously created throughout the study period, all district identifiers were mapped to a stable set of administrative boundaries using a curated transition weights matrix. Where a district split occurred, its crop area and production were reallocated to the historical parent unit using area-weighted cropland overlap. Yield was recomputed as Yield = Production / Area following reallocation, rather than reallocating the yield figure directly, to preserve internal consistency between the three reported quantities. Extreme yield values, arising from administrative reporting errors, were clipped at crop-specific physically plausible thresholds (e.g., 10.0 t/ha for rice and wheat; 250.0 t/ha for sugarcane); the full threshold table is provided in the data dictionary.

The final district-name harmonization pass (documented in full in `district_name_mappings.csv`, Section 3.3) additionally resolves spelling inconsistencies between the yield records and the independently-sourced climate/GEE district boundary layer, which were not always identical even for districts with no administrative history of change.

### 2.2. Google Earth Engine Climate Ingestion

Climate and remote sensing predictors were extracted over Google Earth Engine (GEE):

- **Precipitation**: Monthly precipitation sums aggregated from the UCSB CHIRPS Daily (v2.0) dataset at ~5.5 km resolution.
- **Temperature**: Monthly mean 2-meter air temperature aggregated from the ECMWF ERA5-Land Monthly Aggregated reanalysis dataset (~9 km resolution), converted from Kelvin to Celsius.
- **Soil moisture**: Volumetric soil water content for Layer 1 (0–7 cm) and Layer 2 (7–28 cm), extracted from ERA5-Land.
- **Vegetation indices**: NDVI and EVI bands aggregated from the MODIS MOD13A1 (v061) 16-day composite product at 500 m resolution, scaled by the standard MODIS factor of 10⁻⁴.

To isolate agriculturally relevant weather signal, a spatial cropland mask was applied using the annual MODIS MCD12Q1 (v061) land cover product. Only pixels classified as Croplands (IGBP Class 12) or Cropland/Natural Vegetation Mosaics (IGBP Class 14) in the corresponding year were included in the district-level spatial means, ensuring the reported values reflect conditions over agricultural land rather than urban, forested, or barren terrain.

Vegetation index data for the pre-MODIS era (1997–1999, before MODIS sensor availability from February 2000) were backfilled using the district-level climatological average computed from the observed 2000–2022 record. This period should be treated by users as a synthetic baseline rather than an independent satellite observation, and is flagged accordingly (see Usage Notes, Section 5).

---

## 3. Data Records

The complete dataset is deposited on Zenodo [DOI: 10.5281/zenodo.21126177](https://doi.org/10.5281/zenodo.21126177) as a self-contained folder structure.

### 3.1. Crop Yield CSVs

Fifteen individual files named `crop_{crop_name}_season_year_wide_harmonized.csv` store the area, production, and yield records for each crop. Columns include:

- `Crop_Name`: Crop identifier string.
- `State`, `District`: Harmonized administrative keys, consistent across all files in the dataset.
- `Year`: Harvesting year (1997–2022).
- `{Season}_Area`, `{Season}_Production`, `{Season}_Yield`: Seasonal metrics, where `{Season}` corresponds to the crop's relevant cropping season(s) (Kharif, Rabi, Autumn, or Year-round as applicable). Yield is reported in tonnes/hectare, with the exception of coconut, reported in thousand nuts/hectare.

Missing records (administrative non-reporting) are represented as empty/NaN cells, not sentinel values.

### 3.2. Climate Panel CSV

The wide-format file `district_climate_data_1997_2022.csv` contains monthly climate records for all districts in the harmonized panel. Columns are prefixed by a sequential month index from `0` (January 1997) through `311` (December 2022), where month index *m* for calendar year *Y* and calendar month *M* (1–12) is computed as *m* = (*Y* − 1997) × 12 + (*M* − 1):

- `{m}_Precip_Sum`: Monthly precipitation sum (mm).
- `{m}_Temp_Mean`: Monthly mean 2-meter air temperature (°C).
- `{m}_Soil_L1`, `{m}_Soil_L2`: Volumetric soil moisture, layers 1 and 2 (m³/m³).
- `{m}_NDVI`, `{m}_EVI`: MODIS vegetation indices (computed for the June–September calendar window only, across all districts regardless of locally grown crop (not limited to Kharif-cropped areas); scaled by 0.0001).

### 3.3. Boundary Crosswalk CSV

The file `district_name_mappings.csv` documents the full harmonization ruleset (102 entries):

- `raw_name`: The original district name string as it appears in the source yield files.
- `harmonized_name`: The canonical name used consistently across the yield and climate datasets in this release.
- `state`: The enclosing state or union territory.
- `mapping_type`: Classification of the boundary adjustment — `spelling variant`, `rename`, or `split`. Merges do not exist as a separate category, and are instead handled by reallocating the split child records back into their respective historical parent.
- `parent_district`: For split-type entries, the parent district to which crop statistics were reallocated.
- `notes`: Specific local administrative reorganization context (e.g., date and legislative basis of the boundary change, where known).

### 3.4. Data Dictionary

`data_dictionary.md` provides a complete column-level reference across all files, including units, source datasets, temporal coverage, and the GEE month-indexing convention described in Section 3.2.

---

## 4. Technical Validation

### 4.1. Key Harmonization Validation

To verify boundary mapping correctness, we matched every distinct (State, District) key in the yield datasets against the climate panel's district keys. Prior to the harmonization pass, direct merges using raw district names resulted in match rates between 70.38% and 87.28% depending on the crop, due to unreconciled splits, renames, and spelling inconsistencies between the two independently-sourced datasets. Following harmonization, a plain inner join on `State` and `District` achieves a 100.00% match rate with zero unmatched rows across all 15 crop files.

Row-count integrity was separately verified: total row counts for each yield file are identical before and after the harmonization pass (e.g., 14,990 rows for the rice file in both cases), confirming that no records were lost, duplicated, or double-counted during district reallocation.

Spot checks on individual newly-mapped districts confirm the merged data is physically sensible, not merely key-matched. For example, Leh (Jammu & Kashmir), mapped to the canonical name "Leh (Ladakh)" to align with the climate dataset's post-2019 administrative naming, merges successfully and returns a June mean temperature of −4.75 °C for 2010 — consistent with its high-altitude cold-desert climate. Bajali and Majuli (Assam), both newly created districts reallocated to their historical parents (Barpeta and Jorhat respectively), similarly return non-null, plausible precipitation and temperature values after merging. The Leh mapping was additionally cross-verified for consistency against the independent `clean_india_district_conversion_weights.csv` boundary-weighting table and the `district_neighbors.json` spatial adjacency structure, both of which use the identical canonical naming.

### 4.2. Dataset Usability Validation

As a basic usability check — confirming the joined dataset supports standard downstream modeling rather than reproducing any specific scientific finding — a single gradient-boosted regression model (CatBoost, default hyperparameters, no tuning) was trained on the harmonized Kharif rice panel using an out-of-time chronological train/test split. The model achieves R² = 0.78 (RMSE ≈ 0.43 t/ha), indicating that the joined climate, satellite, and yield fields carry a coherent, learnable predictive relationship consistent with published district-level yield modeling benchmarks for India. This check is intended solely to confirm dataset integrity and usability; detailed model benchmarking, validation-scheme comparisons, and feature engineering are outside the scope of this Data Descriptor and are addressed in the companion modeling study (Chaturvedy & Gururajan, in preparation).

### 4.3. Sentinel and Missing-Value Handling Validation

The original source files encoded missing values using a `-1.0` sentinel. Because all physical quantities in the yield files (area, production, yield) are non-negative by definition, an assertion check was used to confirm that all remaining numerical values in the cleaned files are ≥ 0, verifying that the sentinel-replacement pass (converting `-1.0` to empty/NaN) did not inadvertently alter any legitimate data value.

---

## 5. Usage Notes

- **Merging yield and climate data**: The climate panel is stored in wide format, with each row representing one district and columns representing sequential monthly indices. To join a given crop-year yield record to its corresponding climate variables, users should compute the month index using the formula in Section 3.2, or use the accompanying demo script (`demo_load.py`) as a reference implementation.
- **Pre-2000 vegetation index caveat**: NDVI/EVI values for 1997–1999 are 
  climatological backfills, not observed satellite measurements (Section 2.2). 
  Users conducting time-series or anomaly-based analyses spanning this period 
  should treat it as a synthetic baseline and may wish to exclude these three 
  years, or construct their own binary flag (`Year >= 2000`) to distinguish 
  observed from imputed vegetation index values.
- **Missing shapefile**: The district boundary shapefile (`india_district_administered.geojson`) used for the original GEE spatial aggregation is not included in this release due to size/licensing considerations. Users wishing to reproduce or extend the raw GEE extraction step will need to source an equivalent harmonized district boundary shapefile independently; the boundary *crosswalk* (district name reconciliation) is fully provided and does not require the shapefile to use.
- **Recommended validation scheme for downstream modeling**: Users training predictive models on this panel are encouraged to use out-of-time (chronological) validation rather than random k-fold cross-validation, given the panel's structured spatio-temporal dependencies; see the companion modeling paper for a detailed treatment of this issue.

---

## 6. Code & Data Availability

The full data ingestion, boundary harmonization, and GEE extraction pipeline is available at https://github.com/AadityaChaturvedy/TerraYield-ML. The specific scripts used to generate this release are:

- `district_mappings.py` — canonical district mapping ruleset (source for `district_name_mappings.csv`)
- `data_ingestion/gee_climate_extraction.py` — GEE-based climate and satellite variable extraction (requires user-supplied `GEE_PROJECT` and `GEE_ASSET_ID` environment variables; see repository README)
- `data_ingestion/apply_weights_to_exports.py` — area-weighted boundary reallocation
- `scratch/demo_load.py` — minimal reference example for loading and merging 
  the released dataset (development copy in the source repository; packaged 
  as `demo_load.py` at the root of this Zenodo release)

### Licensing
- **Code**: All processing and pipeline code is licensed under the open-source **MIT License**.
- **Data**: The harmonized dataset panel, standalone crosswalk, and data dictionary are released under the **Creative Commons Attribution 4.0 International License (CC-BY-4.0)**, allowing free reuse and redistribution with appropriate academic attribution.

---

## Acknowledgements

The authors acknowledge the Directorate of Economics and Statistics (DES), Department of Agriculture and Farmers Welfare, Ministry of Agriculture and Farmers Welfare, Government of India, as the primary source of crop area, production, and yield data via the Area, Production, and Yield (APY) Portal (data.desagri.gov.in), released under the Government Open Data License – India (GODL-India).

We also acknowledge the Copernicus Climate Change Service (C3S) Climate Data Store (CDS) for the ERA5-Land reanalysis dataset; the Climate Hazards Center (CHC) at the University of California, Santa Barbara, for the CHIRPS precipitation dataset; and the NASA Land Processes Distributed Active Archive Center (LP DAAC) for the MODIS land cover and vegetation index datasets. This dataset contains modified Copernicus Climate Change Service information [2026].

---

## Author Contributions

A.C. designed and implemented the boundary harmonization pipeline, GEE data extraction, and dataset validation, and wrote the manuscript. B.G. supervised the project, contributed to the validation framework design, and reviewed the final manuscript.

---

## Competing Interests

The authors declare no competing interests.
