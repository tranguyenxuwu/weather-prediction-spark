"""
Preprocess IBTrACS WP — Full Basin (no longitude filter)
========================================================
Creates a lightweight parquet with one row per unique (SID, year, month)
for use as the full-basin monthly ground truth in Phase 5.

Usage:
    conda activate pyspark
    python helpers/preprocess_ibtracs_fullbasin.py
"""

from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, year as F_year, month as F_month, countDistinct

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
CSV_PATH   = str(BASE_DIR / "SPARK__DATA" / "ibtracs.WP.list.v04r01.csv")
OUTPUT_DIR = str(BASE_DIR / "parquet_data" / "ibtracs_fullbasin.parquet")

# ── Spark ─────────────────────────────────────────────────────────────────────
spark = SparkSession.builder.appName("IBTrACS_FullBasin").getOrCreate()

# 1. Load raw CSV (row 2 in the file is units — filter it out)
df = spark.read.option("header", "true").csv(CSV_PATH)
df = df.filter(~col("SID").rlike("^\\s*$"))

# 2. Clean numeric columns
def clean_numeric(c):
    return when(col(c).isin("-999", "-9999", "", " "), None).otherwise(col(c)).cast("float")

df_clean = df.select(
    col("SID"),
    col("ISO_TIME").cast("timestamp").alias("timestamp"),
    clean_numeric("LAT").alias("lat"),
    clean_numeric("LON").alias("lon"),
    col("BASIN"),
)

# 3. Filter: valid coords, lat 0–30N, NO longitude filter (full WP basin)
df_filtered = df_clean.filter(
    (col("lat").isNotNull()) &
    (col("lon").isNotNull()) &
    (col("lat") >= 0) & (col("lat") <= 30)
)

# 4. Extract year/month, deduplicate to one row per storm-month
df_with_ym = (
    df_filtered
    .withColumn("year", F_year("timestamp"))
    .withColumn("month", F_month("timestamp"))
)

# Keep unique (SID, year, month) — a storm active across 2 months gets 2 rows
df_final = df_with_ym.select("SID", "year", "month").dropDuplicates()

# 5. Write parquet
df_final.write.mode("overwrite").parquet(OUTPUT_DIR)

# 6. Verification
total_rows = df_final.count()
n_storms = df_final.select("SID").distinct().count()
year_range = df_final.agg({"year": "min"}).collect()[0][0], df_final.agg({"year": "max"}).collect()[0][0]

print("=" * 60)
print("✅ Full-basin IBTrACS preprocessed!")
print(f"   Output     : {OUTPUT_DIR}")
print(f"   Total rows : {total_rows:,} (unique SID-month combos)")
print(f"   Storms     : {n_storms:,} unique SIDs")
print(f"   Year range : {year_range[0]}–{year_range[1]}")
print("=" * 60)

# Quick monthly summary for recent years
print("\n   Sample monthly counts (2020–2024):")
(
    df_final
    .filter(col("year").between(2020, 2024))
    .groupBy("year", "month")
    .agg(countDistinct("SID").alias("storm_count"))
    .orderBy("year", "month")
    .show(60, truncate=False)
)

spark.stop()
