> [!WARNING]
> **Tài liệu này đã lỗi thời.** Đây là kế hoạch gốc (Climate Analog approach) — đã được thay thế hoàn toàn bằng bottom-up ML pipeline. Xem `docs/train.md` và `models/bottom_up_forecast.py` để biết kiến trúc hiện tại.

---

### PHẦN 1: TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)

**Mục tiêu:** Chuyển đổi tất cả về định dạng **Parquet** (định dạng "ruột" của Spark) để truy vấn nhanh gấp 10-50 lần so với CSV/NetCDF.

#### Bước 1.1: Xử lý NOAA OISST (Nặng nhất - 21GB)

Dữ liệu này đang là toàn cầu. Bạn cần làm 2 việc:

1. **Tính chỉ số El Niño (ONI):** Dùng dữ liệu toàn cầu cắt vùng Nino 3.4 để tính chỉ số này trước.
2. **Cắt vùng Biển Đông:** Để ghép với bão sau này.

* **Action:** Viết script Python dùng `xarray` cắt `lat=[0, 30], lon=[100, 130]`, sau đó lưu thành Parquet.
* **Output:** `oisst_biendong.parquet` (sẽ rất nhẹ).

#### Bước 1.2: Xử lý ERA5 (5.5GB)

Dữ liệu này đã cắt vùng sẵn rồi, nhưng đang ở dạng NetCDF.

* **Action:** Dùng `xarray` convert thẳng sang Parquet.
* **Lưu ý:** Rename các cột `u10`, `v10`, `msl` thành tên dễ hiểu hơn như `wind_u`, `wind_v`, `pressure`.
* **Output:** `era5_biendong.parquet`.

#### Bước 1.3: Làm sạch IBTrACS (CSV)

File CSV chứa rất nhiều nhiễu.

* **Action:** Load vào Spark.
* Filter `BASIN = 'WP'`.
* Thay thế `-999`, `-9999` bằng `null`.
* Cast cột `ISO_TIME` sang kiểu `Timestamp`.
* Chỉ giữ lại các cột quan trọng: `SID`, `NAME`, `ISO_TIME`, `LAT`, `LON`, `WMO_WIND`, `WMO_PRES`.


* **Output:** `storms_clean.parquet`.

---

### PHẦN 2: HỢP NHẤT DỮ LIỆU (SPATIO-TEMPORAL JOIN)

**Mục tiêu:** Tạo ra bảng **Master Dataset**. Đây là bước khó nhất vì phải khớp nối Không gian (Grid) và Thời gian.

**Chiến thuật "Làm tròn tọa độ" (Grid Snapping):**
Cả ERA5 và NOAA đều có độ phân giải **0.25 độ**.

* Bão ở tọa độ lẻ (ví dụ: `12.34`, `109.12`).
* Ta sẽ làm tròn tọa độ bão về lưới gần nhất:
* `Lat_Grid = Round(Lat_Storm * 4) / 4`
* `Lon_Grid = Round(Lon_Storm * 4) / 4`
* *Ví dụ:* `12.34` -> `12.25`.



**Quy trình Join trong Spark:**

1. **Bảng Bão (Left):** Làm tròn `LAT`, `LON` thành `LAT_GRID`, `LON_GRID`.
2. **Bảng Môi trường (Right - ERA5 & NOAA):** Join theo điều kiện:
```sql
ON  Storm.LAT_GRID = Env.LAT
AND Storm.LON_GRID = Env.LON
AND Storm.DATE     = Env.DATE  (hoặc Time gần nhất)

```


3. **Kết quả:** Một bảng to đùng chứa: *Tại thời điểm T, bão X đang ở đâu, gió mạnh bao nhiêu, và nước biển chỗ đó nóng bao nhiêu, áp suất xung quanh thế nào.*

---

### PHẦN 3: PHÂN TÍCH & THUẬT TOÁN (CORE ENGINE)

Chúng ta sẽ làm 2 bài toán chính như đã thảo luận:

#### Bài toán 1: Climate Analog (Tìm năm tương đồng)

* **Bước 1 (Feature Engineering):** Tạo vector đặc trưng cho mỗi năm (từ tháng 1 đến tháng 5 - trước mùa bão).
* Vector 


* **Bước 2 (Similarity):**
* Dùng Spark tính khoảng cách **Euclidean** giữa Vector năm nay (2025/2026) với tất cả các năm quá khứ (1980-2024).


* **Bước 3 (Ranking):** Lấy ra Top 3 năm có khoảng cách nhỏ nhất. (Ví dụ: Năm 2026 giống năm 2010 nhất).

#### Bài toán 2: Thống kê mô tả (Descriptive Statistics)

* Vẽ biểu đồ tương quan (Correlation): Trục X là `SST` (Nhiệt độ biển), Trục Y là `WMO_WIND` (Sức gió bão).
* Mục đích: Chứng minh luận điểm *"Nước biển càng nóng, bão càng mạnh"*.

---

### PHẦN 4: VISUALIZATION (DASHBOARD)

Dùng **Streamlit** để dựng Web App báo cáo.

* **Tab 1 - Overview:** Bản đồ nhiệt (Heatmap) trung bình nhiều năm của Biển Đông.
* **Tab 2 - Storm Viewer:** Chọn một cơn bão lịch sử, vẽ đường đi của nó đè lên bản đồ nhiệt độ biển (SST) để thấy bão "hút" nhiệt như thế nào.
* **Tab 3 - Forecast:**
* Chọn "Năm hiện tại".
* Hệ thống hiển thị: "Năm nay giống năm 1998 (90%), năm 2010 (85%)".
* Hiển thị biểu đồ số lượng bão của các năm tương tự đó để người dùng tham khảo.



---

### MÃ NGUỒN MẪU (Bắt đầu ngay với bước 1)

Bạn tạo file `convert_data.py` để xử lý đống NetCDF sang Parquet trước nhé. Đây là code mẫu cho ERA5:

```python
import xarray as xr
import pandas as pd
import os

# Cấu hình
INPUT_DIR = "era5_data"
OUTPUT_DIR = "processed_data/era5"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# Lấy danh sách file
files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.nc')]

for f in files:
    print(f"Processing {f}...")
    
    # 1. Đọc NetCDF bằng Xarray
    ds = xr.open_dataset(os.path.join(INPUT_DIR, f))
    
    # 2. Rename cho dễ hiểu
    ds = ds.rename({
        'u10': 'wind_u',
        'v10': 'wind_v',
        'msl': 'pressure'
    })
    
    # 3. Chuyển sang DataFrame (Pandas)
    # Lưu ý: ERA5 của bạn đã cắt vùng nên to_dataframe() sẽ không bị tràn RAM đâu
    df = ds.to_dataframe().reset_index()
    
    # 4. Lưu sang Parquet (Partition theo Year/Month nếu cần, nhưng file nhỏ thì thôi)
    # Xóa cột 'number', 'expver' nếu thấy thừa
    if 'expver' in df.columns: df = df.drop(columns=['expver', 'number'])
    
    out_name = f.replace('.nc', '.parquet')
    df.to_parquet(os.path.join(OUTPUT_DIR, out_name), index=False)

print("✅ Xong ERA5!")
