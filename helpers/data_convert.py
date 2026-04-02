"""
NetCDF to Spark Parquet Converter
=================================
Converts .nc (NetCDF) files from SPARK__DATA to Parquet format for faster access.

Usage:
    conda activate pyspark
    python data_convert.py

Data sources:
    - ERA5 storms data: u10, v10, msl (wind components and mean sea level pressure)
    - NOAA SST data: Sea surface temperature daily means (1981-2026)
"""

import os
import sys
import gc
import glob
import shutil
import logging
import concurrent.futures
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from pyspark.sql import SparkSession
from pyspark.sql.functions import year as spark_year, month as spark_month, col, lit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "SPARK__DATA"
OUTPUT_DIR = BASE_DIR / "parquet_data"

# Spark configuration - reduced memory for stability
SPARK_CONFIG = {
    "spark.driver.memory": "4g",
    "spark.executor.memory": "2g",
    "spark.sql.parquet.compression.codec": "snappy",
    "spark.sql.shuffle.partitions": "8",  # Increased slightly for parallel tasks
    "spark.driver.maxResultSize": "1g",
}

# Processing configuration
CHUNK_SIZE = 5  # Number of time steps per chunk (reduced to minimize task size warnings)
SPATIAL_CHUNK_SIZE = 180  # Latitude chunks for large files
MAX_WORKERS = 3  # Thread pool workers (conservative to avoid OOM)


def create_spark_session(app_name: str = "NC_to_Parquet_Converter") -> SparkSession:
    """Create and configure a Spark session."""
    builder = SparkSession.builder.appName(app_name)
    
    for key, value in SPARK_CONFIG.items():
        builder = builder.config(key, value)
    
    # Enable Arrow for better pandas to spark conversion performance
    builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
    
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def force_gc():
    """Force garbage collection to free memory."""
    gc.collect()


def clean_directory(path: Path):
    """Remove a directory and recreate it."""
    if path.exists():
        logger.info(f"Cleaning output directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def netcdf_to_parquet_chunked(
    spark: SparkSession,
    nc_path: Path,
    output_path: Path,
    write_mode: str = "append",
    year_value: int = None,
    pbar_pos: int = 0
) -> int:
    """
    Convert a NetCDF file to Parquet using chunked processing.
    
    Processes the file in small chunks to avoid memory issues.
    Each chunk is written to Parquet immediately.
    
    Args:
        spark: Spark session
        nc_path: Path to the NetCDF file
        output_path: Output Parquet path
        write_mode: "overwrite" or "append"
        year_value: Year value to add as column (for SST files)
        pbar_pos: Position for tqdm progress bar
    
    Returns:
        Total number of rows written
    """
    # logger.info(f"Processing: {nc_path.name}")
    
    total_rows = 0
    
    try:
        with xr.open_dataset(nc_path) as ds:
            # Get coordinate names (handle different naming conventions)
            time_dim = 'time' if 'time' in ds.dims else 'valid_time'
            lat_dim = 'lat' if 'lat' in ds.dims else 'latitude'
            lon_dim = 'lon' if 'lon' in ds.dims else 'longitude'
            
            # Get data variables
            data_vars = list(ds.data_vars)
            
            # Get dimension sizes
            n_time = ds.sizes.get(time_dim, 1)
            
            # Process in time chunks
            # Use tqdm for inner loop progress if needed, or just log
            chunks = list(range(0, n_time, CHUNK_SIZE))
            
            for time_start in chunks:
                time_end = min(time_start + CHUNK_SIZE, n_time)
                
                # Slice the dataset by time
                time_slice = slice(time_start, time_end)
                ds_chunk = ds.isel({time_dim: time_slice})
                
                # Convert to dataframe
                df = ds_chunk.to_dataframe().reset_index()
                
                # Standardize column names
                rename_map = {}
                if time_dim != 'time' and time_dim in df.columns:
                    rename_map[time_dim] = 'time'
                if lat_dim != 'lat' and lat_dim in df.columns:
                    rename_map[lat_dim] = 'lat'
                if lon_dim != 'lon' and lon_dim in df.columns:
                    rename_map[lon_dim] = 'lon'
                
                if rename_map:
                    df = df.rename(columns=rename_map)
                
                # Keep only needed columns
                keep_cols = ['time', 'lat', 'lon'] + data_vars
                df = df[[c for c in keep_cols if c in df.columns]]
                
                # Drop NaN rows
                df = df.dropna()
                
                if len(df) == 0:
                    continue
                
                # Add year column if provided
                if year_value is not None:
                    df['year'] = year_value
                
                # Convert to Spark DataFrame
                try:
                    spark_df = spark.createDataFrame(df)
                except Exception as e:
                    logger.error(f"Error creating DataFrame for {nc_path.name}: {e}")
                    del df
                    gc.collect()
                    continue
                
                # Add month for partitioning
                spark_df = spark_df.withColumn("month", spark_month(col("time")))
                
                # If year not already added, extract from time
                if year_value is None:
                    spark_df = spark_df.withColumn("year", spark_year(col("time")))
                
                # Write to Parquet (Always append since we cleared dir at start)
                spark_df.write.mode("append").partitionBy("year", "month").parquet(str(output_path))
                
                total_rows += len(df)
                
                # Clear memory
                del df, spark_df
                
                if time_start % (CHUNK_SIZE * 5) == 0:
                     force_gc()
            
    except Exception as e:
        logger.error(f"Failed to process {nc_path.name}: {e}")
        return 0

    return total_rows


def process_file_wrapper(args):
    """Wrapper function for thread pool execution."""
    spark, nc_file, output_path, idx, total_files = args
    
    # Calculate years locally if needed for SST
    year_val = None
    if "sst" in str(output_path):
        try:
             year_val = int(nc_file.stem.split('.')[-1])
        except:
             pass

    rows = netcdf_to_parquet_chunked(
        spark, 
        nc_file, 
        output_path, 
        write_mode="append", 
        year_value=year_val,
        pbar_pos=idx
    )
    force_gc()
    return rows


def convert_dataset_parallel(spark: SparkSession, name: str, source_dir: Path, output_path: Path):
    """Convert a dataset using parallel processing."""
    nc_files = sorted(list(source_dir.glob("*.nc")))
    if not nc_files:
        logger.warning(f"No {name} NetCDF files found in {source_dir}")
        return

    logger.info(f"Starting conversion for {name} ({len(nc_files)} files)...")
    
    # Clean output first to ensure fresh start
    clean_directory(output_path)
    
    total_rows_written = 0
    
    # Prepare arguments for map
    # Note: SparkSession is thread-safe for scheduling, but heavy operations might contend.
    # However, since we are doing heavy Python-side processing with xarray/pandas before Spark, 
    # the GIL might be a bottleneck, but I/O and Spark submit should benefit.
    
    tasks = [(spark, f, output_path, i, len(nc_files)) for i, f in enumerate(nc_files)]
    
    with tqdm(total=len(nc_files), desc=f"{name} Progress", unit="file") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(process_file_wrapper, task): task[1].name for task in tasks}
            
            for future in concurrent.futures.as_completed(future_to_file):
                fname = future_to_file[future]
                try:
                    rows = future.result()
                    total_rows_written += rows
                    # Update progress bar description with latest file info
                    pbar.set_postfix({"last_file": fname, "rows": f"{rows:,}"})
                except Exception as e:
                    logger.error(f"Error converting {fname}: {e}")
                finally:
                    pbar.update(1)
                    force_gc()
                    
    logger.info(f"{name} conversion complete! Total rows: {total_rows_written:,}")


def convert_era5_data(spark: SparkSession) -> None:
    """Convert ERA5 storms data to Parquet."""
    era5_dir = DATA_DIR / "era5_data"
    output_path = OUTPUT_DIR / "era5_storms"
    convert_dataset_parallel(spark, "ERA5", era5_dir, output_path)


def convert_noaa_sst_data(spark: SparkSession) -> None:
    """Convert NOAA SST data to Parquet."""
    sst_dir = DATA_DIR / "noaa_sst_data"
    output_path = OUTPUT_DIR / "noaa_sst"
    convert_dataset_parallel(spark, "NOAA SST", sst_dir, output_path)


def verify_parquet_data(spark: SparkSession) -> None:
    """Verify the converted Parquet data."""
    logger.info("\n=== Verifying Parquet Data ===")
    
    # Check ERA5 data
    era5_path = OUTPUT_DIR / "era5_storms"
    if era5_path.exists():
        try:
            era5_df = spark.read.parquet(str(era5_path))
            count = era5_df.count()
            logger.info(f"\nERA5 Storms Data:")
            logger.info(f"  Total rows: {count:,}")
            logger.info(f"  Schema snippet: {era5_df.columns[:5]}...")
            era5_df.show(3)
        except Exception as e:
            logger.error(f"Failed to read ERA5 parquet: {e}")
        finally:
             if 'era5_df' in locals(): del era5_df
             force_gc()
    
    # Check NOAA SST data
    sst_path = OUTPUT_DIR / "noaa_sst"
    if sst_path.exists():
        try:
            sst_df = spark.read.parquet(str(sst_path))
            count = sst_df.count()
            logger.info(f"\nNOAA SST Data:")
            logger.info(f"  Total rows: {count:,}")
            sst_df.show(3)
        except Exception as e:
            logger.error(f"Failed to read SST parquet: {e}")
        finally:
            if 'sst_df' in locals(): del sst_df
            force_gc()


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("NetCDF to Spark Parquet Converter (Optimized)")
    logger.info("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize Spark
    logger.info("\nInitializing Spark session...")
    spark = create_spark_session()
    
    try:
        # Convert ERA5 data
        convert_era5_data(spark)
        force_gc()
        
        # Convert NOAA SST data
        convert_noaa_sst_data(spark)
        force_gc()
        
        # Verify the data
        verify_parquet_data(spark)
        
        logger.info("\n" + "=" * 60)
        logger.info("All conversions completed successfully!")
        logger.info(f"Output directory: {OUTPUT_DIR}")
        logger.info("=" * 60)
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
