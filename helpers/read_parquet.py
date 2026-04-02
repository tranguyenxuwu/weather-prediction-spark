import logging
import sys
from pathlib import Path
from pyspark.sql import SparkSession

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
PARQUET_DIR = BASE_DIR / "parquet_data"

# Spark configuration (matching convert_sst.py for consistency)
SPARK_CONFIG = {
    "spark.driver.memory": "4g",
    "spark.executor.memory": "2g",
    "spark.sql.parquet.compression.codec": "snappy",
    "spark.sql.shuffle.partitions": "8",
    "spark.driver.maxResultSize": "1g",
}

def create_spark_session(app_name: str = "Parquetreader") -> SparkSession:
    """Create and configure a Spark session."""
    builder = SparkSession.builder.appName(app_name)
    
    for key, value in SPARK_CONFIG.items():
        builder = builder.config(key, value)
    
    # Enable Arrow for better performance if needed, though mostly for pandas conversion
    builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
    
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def inspect_parquet_data(spark: SparkSession, data_dir: Path):
    """
    Recursively find parquet datasets (directories containing parquet files) 
    and inspect them.
    """
    if not data_dir.exists():
        logger.error(f"Directory not found: {data_dir}")
        return

    # Check immediate subdirectories for potential datasets
    # Logic: if a directory contains .parquet files directly, it's a dataset.
    # If it contains subdirectories like "year=XXXX", it is a partitioned dataset (which Spark handles automatically at the root).
    
    # We will simply look for the top-level subdirectories in parquet_data
    # e.g., 'noaa_sst', 'era5_storms'
    
    subdirs = [x for x in data_dir.iterdir() if x.is_dir()]
    
    if not subdirs:
        logger.info(f"No subdirectories found in {data_dir}. Checking if it is a dataset itself...")
        subdirs = [data_dir]

    for subdir in subdirs:
        try:
            logger.info(f"--- Inspecting: {subdir.name} ---")
            
            # Read the parquet data
            # Spark automatically handles partitions like year=XXXX if we read the root
            df = spark.read.parquet(str(subdir))
            
            # Print Info
            count = df.count()
            logger.info(f"Row Count: {count:,}")
            
            logger.info("Schema:")
            df.printSchema()
            
            logger.info("Sample Data (Top 20):")
            df.show(20, truncate=False)
            
        except Exception as e:
            # It might not be a parquet dataset
            logger.warning(f"Could not read {subdir.name} as parquet (or it is empty/invalid): {e}")

def main():
    logger.info("Starting Parquet Inspector")
    
    if not PARQUET_DIR.exists():
        logger.error(f"Parquet data directory does not exist at: {PARQUET_DIR}")
        sys.exit(1)
        
    spark = create_spark_session()
    
    try:
        inspect_parquet_data(spark, PARQUET_DIR)
    finally:
        spark.stop()
        logger.info("Spark session stopped.")

if __name__ == "__main__":
    main()
