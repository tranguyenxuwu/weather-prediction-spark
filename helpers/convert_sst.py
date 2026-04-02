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
    level=logging.ERROR,  # Reduced log level to avoid spam
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "SPARK__DATA"
OUTPUT_DIR = BASE_DIR / "parquet_data"

# Spark configuration
SPARK_CONFIG = {
    "spark.driver.memory": "4g",
    "spark.executor.memory": "2g",
    "spark.sql.parquet.compression.codec": "snappy",
    "spark.sql.shuffle.partitions": "8",
    "spark.driver.maxResultSize": "1g",
}

# Processing configuration
CHUNK_SIZE = 5
MAX_WORKERS = 3


def create_spark_session(app_name: str = "SST_Converter") -> SparkSession:
    """Create and configure a Spark session."""
    builder = SparkSession.builder.appName(app_name)
    
    for key, value in SPARK_CONFIG.items():
        builder = builder.config(key, value)
    
    builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
    
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def force_gc():
    """Force garbage collection."""
    gc.collect()


def netcdf_to_parquet_chunked(
    spark: SparkSession,
    nc_path: Path,
    output_path: Path,
    year_value: int
) -> int:
    """
    Convert a NetCDF file to Parquet using chunked processing.
    Writes to a specific year directory to avoid partition race conditions.
    """
    total_rows = 0
    
    try:
        with xr.open_dataset(nc_path) as ds:
            time_dim = 'time' if 'time' in ds.dims else 'valid_time'
            
            # Get data variables
            data_vars = list(ds.data_vars)
            n_time = ds.sizes.get(time_dim, 1)
            
            # Process in time chunks
            chunks = list(range(0, n_time, CHUNK_SIZE))
            
            for time_start in chunks:
                time_end = min(time_start + CHUNK_SIZE, n_time)
                
                # Slice logic
                time_slice = slice(time_start, time_end)
                ds_chunk = ds.isel({time_dim: time_slice})
                
                df = ds_chunk.to_dataframe().reset_index()
                
                # Standardize columns
                rename_map = {}
                if time_dim != 'time' and time_dim in df.columns: rename_map[time_dim] = 'time'
                if 'latitude' in df.columns: rename_map['latitude'] = 'lat'
                if 'longitude' in df.columns: rename_map['longitude'] = 'lon'
                
                if rename_map:
                    df = df.rename(columns=rename_map)
                
                keep_cols = ['time', 'lat', 'lon'] + data_vars
                df = df[[c for c in keep_cols if c in df.columns]]
                df = df.dropna()
                
                if len(df) == 0:
                    continue
                
                # Create Spark DataFrame
                try:
                    spark_df = spark.createDataFrame(df)
                    
                    # Add month for partitioning
                    spark_df = spark_df.withColumn("month", spark_month(col("time")))
                    
                    # We DO NOT add "year" column here because it is implied by the output path
                    # writing to .../year=1986/ which acts as the partition key.
                    # However, to be rigorously safe and allow "partitionBy" to work as expected
                    # without creating nested `year=1986/year=1986`, we rely on Spark's discovery.
                    
                    # Write to unique year path: .../noaa_sst/year=XXXX
                    # partitionBy ONLY "month"
                    
                    spark_df.write.mode("append").partitionBy("month").parquet(str(output_path))
                    
                    total_rows += len(df)
                except Exception as e:
                    logger.error(f"Error in chunk {time_start}: {e}")
                finally:
                    del df
                    if 'spark_df' in locals(): del spark_df
                
                if time_start % (CHUNK_SIZE * 5) == 0:
                     force_gc()
            
    except Exception as e:
        logger.error(f"Failed to process {nc_path.name}: {e}")
        return 0

    return total_rows


def process_file_wrapper(args):
    """Wrapper for thread pool."""
    spark, nc_file, base_output_dir = args
    
    # Extract year
    try:
        year = int(nc_file.stem.split('.')[-1])
        # Unique output path for this file/year to avoid lock contention
        # usage: .../noaa_sst/year=1986
        output_path = base_output_dir / f"year={year}"
        
        # Ensure the directory is clean if it's the first time?
        # Since we use "append", and we might re-run, we should ideally clean it BEFORE the loop.
        # But cleaning specific year dir inside a thread is safe if threads have unique years.
        
        # Note: 'spark' object usage is thread-safe for submission but we rarely want to
        # do DDL (like dropping tables) concurrently.
        # mkdir is fine.
        output_path.mkdir(parents=True, exist_ok=True)
        
        rows = netcdf_to_parquet_chunked(spark, nc_file, output_path, year)
        return rows
    except Exception as e:
        logger.error(f"Error in wrapper for {nc_file}: {e}")
        return 0
    finally:
        force_gc()


def main():
    logger.info("Starting NOAA SST Data Conversion (Isolated Writes)")
    
    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sst_output = OUTPUT_DIR / "noaa_sst"
    
    # Clean fully if starting fresh?
    # For now, let's assume user wants to overwrite if they run this script
    if sst_output.exists():
        logger.info("Cleaning previous SST data...")
        shutil.rmtree(sst_output)
    sst_output.mkdir(parents=True, exist_ok=True)
    
    spark = create_spark_session()
    
    sst_dir = DATA_DIR / "noaa_sst_data"
    nc_files = sorted(list(sst_dir.glob("*.nc")))
    
    logger.info(f"Found {len(nc_files)} files.")
    
    total_rows = 0
    tasks = [(spark, f, sst_output) for f in nc_files]
    
    with tqdm(total=len(nc_files), desc="SST Conversion") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_file_wrapper, t) for t in tasks]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    rows = future.result()
                    total_rows += rows
                except Exception as e:
                    logger.error(f"Future execution failed: {e}")
                finally:
                    pbar.update(1)
                    force_gc()
    
    logger.info(f"Done! Total rows: {total_rows:,}")
    spark.stop()


if __name__ == "__main__":
    main()
