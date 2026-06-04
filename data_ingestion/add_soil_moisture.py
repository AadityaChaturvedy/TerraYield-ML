import ee
import geopandas as gpd
import geemap
import pandas as pd
import numpy as np
import sys

def main():
    print("Initializing Google Earth Engine...")
    try:
        ee.Initialize()
    except Exception as e:
        print(f"Error initializing GEE: {e}")
        sys.exit(1)

    geojson_path = 'data/raw/india_district_administered.geojson'
    csv_path = 'data/processed/district_climate_data_1997_2022.csv'

    print(f"Loading district boundaries from {geojson_path}...")
    gdf = gpd.read_file(geojson_path)
    # Maintain naming columns and geometry
    gdf = gdf[['NAME_1', 'NAME_2', 'geometry']].copy()
    
    # Simplify geometries to speed up transfer to GEE
    print("Simplifying geometries...")
    gdf['geometry'] = gdf['geometry'].simplify(0.01, preserve_topology=True)
    
    print("Converting GeoDataFrame to GEE FeatureCollection...")
    districts_fc = geemap.geopandas_to_ee(gdf)

    print("Loading ERA5-Land monthly soil moisture collection...")
    # Load volumetric soil water layer 1 (0-7cm) and layer 2 (7-28cm)
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
             .filterDate('1997-01-01', '2023-01-01') \
             .select(['volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2'])

    print("Constructing multi-band GEE image for all 312 months...")
    # We build a list of monthly images and concatenate their bands
    total_months = 26 * 12
    
    # Pre-build local images list to avoid mapping ee.List overhead
    months_images = []
    start_date = ee.Date('1997-01-01')
    for m in range(total_months):
        start = start_date.advance(m, 'month')
        end = start.advance(1, 'month')
        img = era5.filterDate(start, end).mean()
        # Rename bands to {m}_Soil_L1 and {m}_Soil_L2
        img_renamed = img.select(['volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2']) \
                         .rename([f"{m}_Soil_L1", f"{m}_Soil_L2"])
        months_images.append(img_renamed)
    
    # Concatenate all bands into a single ee.Image
    combined_img = ee.Image.cat(months_images)

    print("Reducing regions (calculating spatial means for each district in GEE)...")
    reduced = combined_img.reduceRegions(
        collection=districts_fc,
        reducer=ee.Reducer.mean(),
        scale=9000, # ERA5-Land resolution is ~9km
        crs='EPSG:4326'
    )

    print("Requesting and downloading reduced soil moisture data from GEE...")
    # Fetch data as list of features
    try:
        features = reduced.getInfo()['features']
    except Exception as e:
        print(f"Error fetching data from GEE: {e}")
        sys.exit(1)

    print(f"Extracted features for {len(features)} districts. Processing results...")
    rows = []
    for f in features:
        props = f['properties']
        rows.append(props)

    df_soil = pd.DataFrame(rows)
    
    # Load existing CSV
    print(f"Loading existing climate CSV from {csv_path}...")
    df_existing = pd.read_csv(csv_path)

    # Merge by NAME_1 and NAME_2
    print("Merging soil moisture columns with existing climate data...")
    # Drop any GEE system columns from df_soil if they exist to keep it clean
    if 'system:index' in df_soil.columns:
        df_soil.drop(columns=['system:index'], inplace=True)
        
    df_merged = pd.merge(df_existing, df_soil, on=['NAME_1', 'NAME_2'], how='left')

    print(f"Saving merged data back to {csv_path}...")
    df_merged.to_csv(csv_path, index=False)
    print("Soil moisture successfully integrated!")

if __name__ == "__main__":
    main()
