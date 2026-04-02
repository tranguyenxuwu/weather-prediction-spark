"""
Part 2: Spatio-Temporal Join — Master Dataset
==============================================
Merges ERA5 (atmosphere), NOAA SST (ocean), ONI (ENSO), and IBTrACS (storms)
into one environment-centric master dataset.

Processes YEAR-BY-YEAR to avoid filling up disk with shuffle data.

Usage:
    conda activate pyspark
    python spatio_temporal_join.py

Output:
    parquet_data/master_dataset.parquet/
"""

import logging
import sys
import time
import shutil
from pathlib import Path

from tqdm import tqdm

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# ── Config ────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("SpatioTemporalJoin")

BASE_DIR = Path(__file__).resolve().parent.parent   # WeatherPredict/
PARQUET_DIR = BASE_DIR / "parquet_data"

ERA5_DIR    = PARQUET_DIR / "era5"
SST_PATH    = str(PARQUET_DIR / "noaa_sst")
STORMS_PATH = str(PARQUET_DIR / "ibtracs_clean.parquet")
ONI_PATH    = str(BASE_DIR / "oni.csv")
OUTPUT_PATH = str(PARQUET_DIR / "master_dataset.parquet")

# Use external SSD for Spark temp to avoid filling macOS /tmp
SPARK_LOCAL_DIR = str(BASE_DIR / ".spark_tmp")

KTS_TO_KMH = 1.852

SPARK_CONFIG = {
    "spark.driver.memory": "10g",
    "spark.executor.memory": "6g",
    "spark.sql.shuffle.partitions": "16",
    "spark.sql.parquet.compression.codec": "snappy",
    "spark.driver.maxResultSize": "2g",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.execution.arrow.pyspark.enabled": "true",
    # Pandas writes timestamps as TIMESTAMP(NANOS) — PySpark 4.x can't read these.
    "spark.sql.legacy.parquet.nanosAsLong": "true",
    # Redirect temp/shuffle to external SSD
    "spark.local.dir": SPARK_LOCAL_DIR,
    # Show Spark jobs progress bar in console
    "spark.ui.showConsoleProgress": "true",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def create_spark():
    builder = SparkSession.builder.appName("SpatioTemporalJoin")
    for k, v in SPARK_CONFIG.items():
        builder = builder.config(k, v)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def nanos_to_date(col_name):
    """Convert a nanosecond-epoch long column to a date column."""
    return F.to_date(F.from_unixtime(F.col(col_name) / 1_000_000_000))


# ── Load ONI (small — load once) ─────────────────────────────────────────────

def load_oni(spark):
    """Load ONI CSV, classify ENSO phase as 0/1/2."""
    log.info("Loading ONI data...")

    oni_raw = spark.read.csv(ONI_PATH, header=True)
    cols = oni_raw.columns

    oni = (
        oni_raw
        .withColumnRenamed(cols[0], "raw_date")
        .withColumnRenamed(cols[1], "raw_oni")
    )

    oni = (
        oni
        .withColumn("oni_date", F.to_date(F.trim(F.col("raw_date"))))
        .withColumn("oni_value", F.trim(F.col("raw_oni")).cast("float"))
    )

    # Replace -9999 with null
    oni = oni.withColumn(
        "oni_value",
        F.when(F.col("oni_value") < -999, None).otherwise(F.col("oni_value"))
    )

    # 0=Neutral, 1=El Niño, 2=La Niña
    oni = oni.withColumn(
        "enso_phase",
        F.when(F.col("oni_value") >= 0.5, F.lit(1))
         .when(F.col("oni_value") <= -0.5, F.lit(2))
         .otherwise(F.lit(0))
         .cast(IntegerType())
    )

    oni = (
        oni
        .withColumn("oni_year", F.year("oni_date"))
        .withColumn("oni_month", F.month("oni_date"))
        .select("oni_year", "oni_month", "oni_value", "enso_phase")
    )

    log.info(f"   ONI: {oni.count()} monthly records loaded.")
    return oni


# ── Load Storms (small — load once) ──────────────────────────────────────────

def load_storms(spark):
    """Load IBTrACS, grid-snap, convert wind to km/h."""
    log.info("Loading IBTrACS storm data...")

    storms = spark.read.parquet(STORMS_PATH)

    storms = (
        storms
        .withColumn("lat_grid", F.round(F.col("lat") * 4) / 4)
        .withColumn("lon_grid", F.round(F.col("lon") * 4) / 4)
        .withColumn("storm_date", F.to_date(F.col("timestamp")))
        .withColumn("wind_speed_kmh", F.round(F.col("wind_speed_wmo") * KTS_TO_KMH, 1))
    )

    storms = storms.select(
        F.col("lat_grid").alias("storm_lat"),
        F.col("lon_grid").alias("storm_lon"),
        "storm_date",
        "SID", "NAME", "wind_speed_kmh", "pressure_wmo",
    )

    log.info(f"   Storms: {storms.count()} track points loaded.")
    return storms


# ── Process One Year ──────────────────────────────────────────────────────────

def process_year(spark, year, oni, storms, write_mode="append"):
    """Process a single year: ERA5 daily + SST + ONI + Storms → master."""

    era5_file = ERA5_DIR / f"era5_merged_{year}.parquet"
    if not era5_file.exists():
        tqdm.write(f"   [WARNING] No ERA5 file for {year}, skipping.")
        return 0

    t0 = time.time()

    # ── ERA5 daily aggregation ──
    era5 = spark.read.parquet(str(era5_file))
    era5 = (
        era5
        .withColumnRenamed("latitude", "lat")
        .withColumnRenamed("longitude", "lon")
    )
    time_col = "valid_time" if "valid_time" in era5.columns else "time"
    era5 = era5.withColumn("date", nanos_to_date(time_col))

    era5_daily = (
        era5
        .groupBy("lat", "lon", "date")
        .agg(
            F.avg("u_wind").alias("u_wind_avg"),
            F.avg("v_wind").alias("v_wind_avg"),
            F.avg("slp").alias("slp_avg"),
            F.avg("wind_speed_env").alias("wind_speed_env_avg"),
        )
    )

    # ── SST for this year ──
    sst_year_path = PARQUET_DIR / "noaa_sst" / f"year={year}"
    if sst_year_path.exists():
        sst = spark.read.parquet(str(sst_year_path))

        # ── FIX: SST grid is offset 0.125° from ERA5 (center vs corner).
        #    Snap SST lat/lon to the ERA5 0.25° grid so the join matches.
        sst = sst.withColumn("lat", F.round(F.col("lat") * 4) / 4)
        sst = sst.withColumn("lon", F.round(F.col("lon") * 4) / 4)

        # ── FIX: SST is global — filter to ERA5 region to avoid shuffling
        #    the entire ocean. ERA5 covers lat [0, 30], lon [100, 180].
        sst = sst.filter(
            (F.col("lat") >= 0) & (F.col("lat") <= 30) &
            (F.col("lon") >= 100) & (F.col("lon") <= 180)
        )

        sst_time_col = "time" if "time" in sst.columns else "valid_time"
        col_type = dict(sst.dtypes).get(sst_time_col, "")
        if "long" in col_type or "bigint" in col_type:
            sst = sst.withColumn("date", nanos_to_date(sst_time_col))
        else:
            sst = sst.withColumn("date", F.to_date(F.col(sst_time_col)))

        sst_daily = (
            sst.groupBy("lat", "lon", "date")
            .agg(F.avg("sst").alias("sst_avg"))
        )
    else:
        sst_daily = None

    # ── Join ERA5 + SST ──
    if sst_daily is not None:
        master = era5_daily.join(sst_daily, on=["lat", "lon", "date"], how="left")
    else:
        master = era5_daily.withColumn("sst_avg", F.lit(None).cast("double"))

    # ── Join ONI ──
    master = (
        master
        .withColumn("year", F.year("date"))
        .withColumn("month", F.month("date"))
    )
    master = master.join(
        oni,
        on=[master["year"] == oni["oni_year"], master["month"] == oni["oni_month"]],
        how="left"
    ).drop("oni_year", "oni_month")

    # ── Join Storms ──
    master = master.join(
        storms,
        on=[
            master["lat"] == storms["storm_lat"],
            master["lon"] == storms["storm_lon"],
            master["date"] == storms["storm_date"],
        ],
        how="left"
    ).drop("storm_lat", "storm_lon", "storm_date")

    # ── Select final columns ──
    master = master.select(
        "lat", "lon", "date", "year", "month",
        "u_wind_avg", "v_wind_avg", "slp_avg", "wind_speed_env_avg",
        "sst_avg",
        "oni_value", "enso_phase",
        "SID", "NAME", "wind_speed_kmh", "pressure_wmo",
    )

    # ── Write ──
    master.write.mode(write_mode).parquet(OUTPUT_PATH)

    elapsed = time.time() - t0
    tqdm.write(f"   Year {year} written in {elapsed:.0f}s")
    return 1


# ── Verification ──────────────────────────────────────────────────────────────

def verify(spark):
    """Print verification stats for the master dataset."""
    log.info("=" * 60)
    log.info("VERIFICATION")
    log.info("=" * 60)

    result = spark.read.parquet(OUTPUT_PATH)
    total = result.count()
    log.info(f"Total rows: {total:,}")
    result.printSchema()

    # Storm coverage
    storm_rows = result.filter(F.col("SID").isNotNull()).count()
    log.info(f"Rows WITH storm: {storm_rows:,} ({storm_rows / total * 100:.4f}%)")
    log.info(f"Rows WITHOUT storm: {total - storm_rows:,}")

    # Null rates
    log.info("\nNull rates:")
    for c in tqdm(["sst_avg", "u_wind_avg", "oni_value", "SID"], desc="Checking null rates"):
        n = result.filter(F.col(c).isNull()).count()
        tqdm.write(f"  {c}: {n:,} ({n / total * 100:.2f}%)")

    # ENSO distribution
    log.info("\nENSO phase distribution:")
    result.groupBy("enso_phase").count().orderBy("enso_phase").show()

    # Sample WITH storm
    log.info("Sample rows WITH storms:")
    result.filter(F.col("SID").isNotNull()).show(10, truncate=False)

    # Sample WITHOUT storm
    log.info("Sample rows WITHOUT storms:")
    result.filter(F.col("SID").isNull()).show(5, truncate=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Part 2: Spatio-Temporal Join — Master Dataset")
    log.info("=" * 60)

    # Create Spark temp dir on external SSD
    Path(SPARK_LOCAL_DIR).mkdir(parents=True, exist_ok=True)

    # Clean previous output
    output = Path(OUTPUT_PATH)
    if output.exists():
        log.info("Cleaning previous master dataset...")
        shutil.rmtree(output)

    spark = create_spark()

    try:
        # Load small datasets once
        oni = load_oni(spark)
        oni.cache()  # Small dataset, keep in memory

        storms = load_storms(spark)
        storms.cache()

        # Get year range from ERA5 files
        era5_files = sorted(ERA5_DIR.glob("era5_merged_*.parquet"))
        years = [int(f.stem.split("_")[-1]) for f in era5_files]
        log.info(f"\nProcessing {len(years)} years: {years[0]}\u2013{years[-1]}")

        # Process year by year with progress bar
        pbar = tqdm(years, desc="Building master dataset", unit="year",
                    bar_format="{l_bar}{bar:30}{r_bar}")
        for i, year in enumerate(pbar, 1):
            pbar.set_postfix(year=year, refresh=True)
            mode = "overwrite" if i == 1 else "append"
            process_year(spark, year, oni, storms, write_mode=mode)
        pbar.close()

        # Verify
        verify(spark)

        log.info("\n✅ Master dataset created successfully!")
        log.info(f"   Output: {OUTPUT_PATH}")

    except Exception as e:
        log.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up Spark temp
        spark.stop()
        tmp = Path(SPARK_LOCAL_DIR)
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
