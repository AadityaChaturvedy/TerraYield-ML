import ee
import pandas as pd
import geopandas as gpd
import time
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Time period for the TERRA dataset
START_DATE = '1997-01-01'
END_DATE = '2023-01-01'

# We are using the local GeoJSON containing your harmonized boundaries
LOCAL_GEOJSON_PATH = 'data/raw/india_district_administered.geojson'
OUTPUT_CSV_PATH = 'data/processed/district_climate_data_1997_2022.csv'

# Set this to True to apply a MODIS MCD12Q1-based cropland mask (Type 1 IGBP classes 12 and 14)
# to exclude non-cropland pixels from climate aggregation (the paper's phenology crop mask)
APPLY_CROPLAND_MASK = True

# Set this to True if you uploaded the GeoJSON to GEE Assets (RECOMMENDED for large files)
# If False, the script will try to convert the local GeoJSON directly (might fail if >10MB)
USE_GEE_ASSET = False
# If USE_GEE_ASSET = True, replace this with your GEE asset ID
GEE_ASSET_ID = 'projects/your-project/assets/india_district_administered'

def initialize_gee():
    """Authenticates and initializes Google Earth Engine."""
    try:
        ee.Initialize(project='your-project-id') # Update if you have a specific GCP project
    except ee.EEException:
        print("Authenticating to Google Earth Engine...")
        ee.Authenticate()
        ee.Initialize()

def get_feature_collection():
    """Loads the district boundaries into GEE."""
    if USE_GEE_ASSET:
        print(f"Loading FeatureCollection from GEE Asset: {GEE_ASSET_ID}")
        return ee.FeatureCollection(GEE_ASSET_ID)
    else:
        print(f"Loading local GeoJSON from {LOCAL_GEOJSON_PATH}...")
        # Import geemap only if doing local conversion
        import geemap
        gdf = gpd.read_file(LOCAL_GEOJSON_PATH)
        
        # We will keep just the identifying columns to reduce payload size
        cols_to_keep = ['NAME_1', 'NAME_2'] 
        cols_to_keep = [c for c in cols_to_keep if c in gdf.columns]
        
        # FIX: Simplify the geometry to bypass the 10MB Google Earth Engine payload limit
        # A tolerance of 0.01 degrees (~1km) shrinks the GeoJSON from 22MB to 1.6MB 
        # without affecting the mean climate aggregation materially.
        print("Simplifying geometries to bypass API payload limits...")
        gdf['geometry'] = gdf['geometry'].simplify(0.01, preserve_topology=True)
        
        # Convert to Earth Engine FeatureCollection
        print("Converting GeoDataFrame to Earth Engine FeatureCollection...")
        # This step can take a few minutes for a 23MB file and might hit payload limits.
        # If it fails, please upload the GeoJSON to GEE Assets manually.
        return geemap.geopandas_to_ee(gdf[cols_to_keep + ['geometry']])

def extract_climate_data():
    initialize_gee()
    districts_fc = get_feature_collection()

    print(f"Extracting ERA5-Land (Temperature) and CHIRPS (Precipitation) from {START_DATE} to {END_DATE}...")

    # Load ERA5-Land (Monthly) - 2m Temperature (convert from Kelvin to Celsius)
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
            .filterDate(START_DATE, END_DATE) \
            .select('temperature_2m')

    def celsius(img):
        return img.subtract(273.15).copyProperties(img, ['system:time_start'])
    
    era5_celsius = era5.map(celsius)

    # Load CHIRPS v2 (Monthly Precipitation in mm)
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
            .filterDate(START_DATE, END_DATE)

    # Load MODIS Land Cover MCD12Q1 yearly collection for cropland masking
    mcd12 = ee.ImageCollection("MODIS/061/MCD12Q1")

    # Combine ERA5 and CHIRPS into a single monthly image collection
    def process_month(month_offset):
        start = ee.Date(START_DATE).advance(month_offset, 'month')
        end = start.advance(1, 'month')
        
        temp_mean = era5_celsius.filterDate(start, end).mean().rename('Temp_Mean')
        precip_sum = chirps.filterDate(start, end).sum().rename('Precip_Sum')
        
        if APPLY_CROPLAND_MASK:
            # Get MCD12Q1 Land Cover for the current year
            year = start.get('year')
            # MODIS MCD12Q1 is yearly. Get land cover for the current year or fallback to nearest available (2001)
            lc_year = ee.Image(ee.Algorithms.If(
                year.gte(2001).And(year.lte(2022)),
                mcd12.filterDate(start.format('YYYY-01-01'), start.format('YYYY-12-31')).first(),
                mcd12.filterDate('2001-01-01', '2001-12-31').first()
            )).select('LC_Type1')
            
            # Crop mask: Class 12 = Croplands, Class 14 = Cropland/Natural Vegetation Mosaics
            crop_mask = lc_year.eq(12).Or(lc_year.eq(14))
            
            temp_mean = temp_mean.updateMask(crop_mask)
            precip_sum = precip_sum.updateMask(crop_mask)
            
        return temp_mean.addBands(precip_sum) \
               .set('system:time_start', start.millis()) \
               .set('Year', start.get('year')) \
               .set('Month', start.get('month'))


    # Calculate total months
    total_months = (2023 - 1997) * 12
    months_list = ee.List.sequence(0, total_months - 1)
    
    climate_monthly = ee.ImageCollection.fromImages(months_list.map(process_month))

    print("Submitting Export Task to Google Drive...")
    print("This will process entirely in the cloud and may take 10-30 minutes.")
    
    # We export to Google Drive rather than pulling it directly into the Python script 
    # because extracting 25 years of monthly data for ~700 districts will timeout locally.
    task = ee.batch.Export.table.toDrive(
        collection=climate_monthly.toBands().reduceRegions(
            collection=districts_fc,
            reducer=ee.Reducer.mean(), # Mean spatial aggregation for the district
            scale=5566, # Scale for CHIRPS (~0.05 degrees)
            crs='EPSG:4326',
            tileScale=16
        ),
        description='TERRA_Climate_Extract_1997_2022',
        folder='GEE_Exports',
        fileNamePrefix='district_climate_data_1997_2022',
        fileFormat='CSV'
    )
    
    task.start()
    
    print("Task started! You can check the status at: https://code.earthengine.google.com/tasks")
    print(f"Once finished, download the CSV from your Google Drive and place it at: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    extract_climate_data()
