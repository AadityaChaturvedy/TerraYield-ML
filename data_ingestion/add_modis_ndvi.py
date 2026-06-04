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
    gdf = gdf[['NAME_1', 'NAME_2', 'geometry']].copy()
    
    print("Simplifying geometries...")
    gdf['geometry'] = gdf['geometry'].simplify(0.01, preserve_topology=True)
    
    print("Converting GeoDataFrame to GEE FeatureCollection...")
    districts_fc = geemap.geopandas_to_ee(gdf)

    print("Loading MODIS Vegetation Indices monthly/16-day collection...")
    modis = ee.ImageCollection("MODIS/061/MOD13A1") \
              .filterDate('2000-01-01', '2023-01-01') \
              .select(['NDVI', 'EVI'])

    print("Constructing multi-band GEE image for Kharif months (June-Sept) from 1997 to 2022...")
    months_images = []
    start_date = ee.Date('1997-01-01')
    
    for year in range(1997, 2023):
        offset = (year - 1997) * 12
        # Kharif months: June (5), July (6), August (7), September (8)
        kharif_months = [offset + 5, offset + 6, offset + 7, offset + 8]
        
        for m in kharif_months:
            if year < 2000:
                # Pre-MODIS era (1997-1999): fill with dummy values
                img_scaled = ee.Image.constant([-9999.0, -9999.0]).rename([f"{m}_NDVI", f"{m}_EVI"]).double()
            else:
                start = start_date.advance(m, 'month')
                end = start.advance(1, 'month')
                monthly_coll = modis.filterDate(start, end)
                
                img_scaled = ee.Image(ee.Algorithms.If(
                    monthly_coll.size().gt(0),
                    monthly_coll.mean().select(['NDVI', 'EVI']).multiply(0.0001).rename([f"{m}_NDVI", f"{m}_EVI"]),
                    ee.Image.constant([-9999.0, -9999.0]).rename([f"{m}_NDVI", f"{m}_EVI"])
                ))
            months_images.append(img_scaled)
    
    # Concatenate all bands into a single ee.Image
    combined_img = ee.Image.cat(months_images)

    print("Reducing regions (calculating spatial means for each district in GEE)...")
    reduced = combined_img.reduceRegions(
        collection=districts_fc,
        reducer=ee.Reducer.mean(),
        scale=10000, # 10km scale is memory-safe and runs extremely fast
        crs='EPSG:4326'
    )

    print("Requesting and downloading reduced MODIS data from GEE...")
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

    df_modis = pd.DataFrame(rows)
    
    # Load existing CSV
    print(f"Loading existing climate CSV from {csv_path}...")
    df_existing = pd.read_csv(csv_path)

    # Merge by NAME_1 and NAME_2
    print("Merging MODIS NDVI/EVI columns with existing climate data...")
    if 'system:index' in df_modis.columns:
        df_modis.drop(columns=['system:index'], inplace=True)
        
    # Find columns that are MODIS bands to make sure we don't have duplicates
    modis_cols = [c for c in df_modis.columns if c not in ['NAME_1', 'NAME_2']]
    
    # Remove existing MODIS columns if they exist to allow clean overwriting
    df_existing.drop(columns=[c for c in modis_cols if c in df_existing.columns], errors='ignore', inplace=True)
        
    df_merged = pd.merge(df_existing, df_modis, on=['NAME_1', 'NAME_2'], how='left')

    print(f"Saving merged data back to {csv_path}...")
    df_merged.to_csv(csv_path, index=False)
    print("MODIS NDVI/EVI successfully integrated!")

if __name__ == "__main__":
    main()
