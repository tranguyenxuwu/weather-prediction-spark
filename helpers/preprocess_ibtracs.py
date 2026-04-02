from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, to_timestamp

# Khởi tạo Spark
spark = SparkSession.builder.appName("Preprocess_IBTrACS").getOrCreate()

# 1. Load dữ liệu (header=True)
# Lưu ý: file CSV này dòng 2 là đơn vị, cần filter sau.
df = spark.read.option("header", "true").csv("SPARK__DATA/ibtracs.WP.list.v04r01.csv")

# 2. Lọc bỏ dòng đơn vị (nơi SID chứa khoảng trắng hoặc là header lặp)
df = df.filter(~col("SID").rlike("^\\s*$"))

# 3. Chọn cột cần thiết & Ép kiểu
# -999, -9999, ' ' (rỗng) đều coi là null
def clean_numeric(c):
    return when(col(c).isin("-999", "-9999", "", " "), None).otherwise(col(c)).cast("float")

df_clean = df.select(
    col("SID"),
    col("NAME"),
    col("ISO_TIME").cast("timestamp").alias("timestamp"),  # Quan trọng: Time chuẩn
    clean_numeric("LAT").alias("lat"),
    clean_numeric("LON").alias("lon"),
    clean_numeric("WMO_WIND").alias("wind_speed_wmo"),    # Gió chuẩn WMO (kts)
    clean_numeric("WMO_PRES").alias("pressure_wmo"),      # Áp suất WMO (mb)
    col("BASIN")
)

# 4. Lọc dữ liệu rác & Vùng biển Đông (South China Sea region)
# Bão phải có tọa độ & nằm trong khung lưới nghiên cứu
df_final = df_clean.filter(
    (col("lat").isNotNull()) &
    (col("lon").isNotNull()) &
    (col("lat") >= 0) & (col("lat") <= 30) &
    (col("lon") >= 100) & (col("lon") <= 130)
)

# 5. Lưu Parquet
df_final.write.mode("overwrite").parquet("SPARK__DATA/processed/ibtracs_clean.parquet")

print("✅ Đã xử lý xong IBTrACS!")
print(f"   Số dòng sau khi lọc: {df_final.count()}")

spark.stop()
