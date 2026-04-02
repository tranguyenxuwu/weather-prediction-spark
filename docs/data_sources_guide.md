# 📡 Hướng Dẫn Bổ Sung Dữ Liệu – Mở Rộng Domain ERA5

> **Mục tiêu:** Mở rộng environmental data từ `100–130°E` (hiện tại) → `100–180°E` để cover toàn bộ Western Pacific genesis domain.

---

## Tại sao cần bổ sung?

| Chỉ số | Hiện tại (100–130°E) | Sau mở rộng (100–180°E) |
|--------|---------------------|------------------------|
| Genesis events trong grid | 404 / 1,413 = **28.6%** | ~1,200 / 1,413 = **~85%** |
| Track points có env data | 35,219 / 98,907 = **35.6%** | ~80,000 / 98,907 = **~81%** |
| Grid cells | 121 × 121 = 14,641 | 121 × 321 = 38,841 |

→ **Tăng 3 lần coverage** cho genesis model, track model sẽ có env features cho ~81% track points.

---

## 📊 3 Nguồn Thay Thế CDS

### 1. ✅ Google Cloud ARCO ERA5 (KHUYẾN NGHỊ)

- **Là gì:** Mirror chính thức của ERA5 trên Google Cloud Storage, format Zarr
- **Resolution:** 0.25° (giống CDS gốc)
- **Coverage:** Global, 1940–present, hourly
- **Chi phí:** FREE (Google Public Dataset)
- **Ưu điểm:** Không cần CDS account, tải nhanh, cloud-native format
- **Nhược điểm:** Cần `gcsfs` + `xarray` + `zarr` (pip install)

**Bucket URL:**
```
gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3
```

**Cài đặt:**
```bash
conda activate pyspark
pip install gcsfs zarr xarray dask
```

**Script tải — xem file:** `SPARK__DATA/arco_era5_download.py`

---

### 2. 🌐 Open-Meteo Historical Weather API

- **Là gì:** REST API miễn phí trả về dữ liệu ERA5 theo tọa độ
- **Resolution:** 0.25° (ERA5 backend)
- **Coverage:** Global, 1940–present, hourly
- **Chi phí:** FREE, không cần API key
- **Giới hạn:** 10,000 calls/ngày (mỗi call = 1 location × time range)
- **Ưu điểm:** Không cần cài gì ngoài `requests`, REST đơn giản
- **Nhược điểm:** Chậm cho grid lớn (cần gọi từng điểm), trả về wind_speed + wind_direction thay vì u10/v10

**API URL:**
```
https://archive-api.open-meteo.com/v1/era5
```

**Ví dụ call:**
```bash
curl "https://archive-api.open-meteo.com/v1/era5?latitude=15&longitude=140&start_date=2020-01-01&end_date=2020-12-31&hourly=wind_speed_10m,wind_direction_10m,pressure_msl&models=era5"
```

**Script tải — xem file:** `SPARK__DATA/openmeteo_era5_download.py`

---

### 3. 📦 NCEP GFS 0.25° Analysis (NOAA NOMADS)

- **Là gì:** GFS analysis (0-hour forecast) = quasi-reanalysis từ NOAA
- **Resolution:** 0.25°
- **Coverage:** Global, 2015–present, 6-hourly
- **Chi phí:** FREE
- **Ưu điểm:** Tải trực tiếp từ NOAA, NetCDF/GRIB2 format
- **Nhược điểm:** Chỉ từ 2015 (không có 1983–2014), format GRIB2 phức tạp

**NOMADS URL:**
```
https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl
```

> [!WARNING]
> GFS chỉ có từ 2015. **Không đủ** cho training set (cần 1983–2015). Chỉ dùng nếu cần real-time forecast data.

---

## 🎯 Khuyến Nghị: Dùng Google Cloud ARCO

| Tiêu chí | ARCO ERA5 | Open-Meteo | NCEP GFS |
|----------|-----------|------------|----------|
| Coverage temporal | 1940–now ✅ | 1940–now ✅ | 2015–now ❌ |
| Resolution | 0.25° ✅ | 0.25° ✅ | 0.25° ✅ |
| Variables u10/v10/msl | ✅ trực tiếp | ⚠️ cần convert | ✅ trực tiếp |
| Tốc độ tải | ⚡ Rất nhanh | 🐌 Chậm (REST) | ⚡ Nhanh |
| Cần account? | ❌ Không | ❌ Không | ❌ Không |
| Consistent với data hiện tại | ✅ Cùng ERA5 | ✅ Cùng ERA5 | ❌ Khác dataset |

**→ Dùng ARCO ERA5 để mở rộng domain từ 130°E → 180°E, giữ nguyên format + variables.**

---

## 📐 Thông Số Tải

### Domain mở rộng

```
Existing:  lat [0°N, 30°N], lon [100°E, 130°E]  → 121 × 121 grid
Extension: lat [0°N, 30°N], lon [130.25°E, 180°E] → 121 × 200 grid
```

### Biến cần tải

| Variable | ERA5 name | Phần mở rộng |
|----------|-----------|-------------|
| `u10` | 10m U-wind | 130.25–180°E |
| `v10` | 10m V-wind | 130.25–180°E |
| `msl` | Mean sea level pressure | 130.25–180°E |

### Kích thước ước tính

```
Extension only (130–180°E):
  Per year = 1464 timesteps × 121 lat × 200 lon × 3 vars × 4 bytes ≈ 420 MB
  42 years = ~17.6 GB (NetCDF)
  
Merged (100–180°E):
  Per year ≈ 550 MB → Total ~23 GB
```

### Timeline

```
6-hourly: 00:00, 06:00, 12:00, 18:00 UTC → 1464 steps/year (leap: 1468)
Years: 1983–2024 → 42 files
```

---

## 🔧 Workflow Tổng Thể

```
Bước 1: pip install gcsfs zarr xarray dask
Bước 2: Chạy SPARK__DATA/arco_era5_download.py  → tải extension (130–180°E)
Bước 3: Chạy SPARK__DATA/merge_era5_extension.py → merge với data hiện tại
Bước 4: Cập nhật helpers/preprocess_era5.py      → xử lý domain mới
Bước 5: Chạy lại helpers/spatio_temporal_join.py  → rebuild master_dataset
```

> [!IMPORTANT]
> **Bước 5 tốn nhiều thời gian nhất** (~hours) vì cần re-join 225M+ rows. Chỉ chạy sau khi đã verify data extension OK.

---

## ⚡ Phương Án Nhanh: Track-Only (KHÔNG cần mở rộng)

Nếu chỉ muốn train **track model** mà không cần genesis toàn basin:

1. Dùng **IBTrACS data trực tiếp** (đã có) → 97K track points + WMO_WIND + WMO_PRES
2. Env features = NaN cho track points ngoài 100–130°E → LightGBM tự handle
3. **Không cần tải thêm bất kỳ data nào**

Chỉ cần tải thêm nếu muốn:
- Genesis prediction toàn basin (không chỉ SCS)
- Env features cho track model ngoài 100–130°E (tăng accuracy ~5–10%)
