# WeatherPredict 🌪️

[Vietnamese Version](#phiên-bản-tiếng-việt) | [English Version](#english-version)

---

### Project Summary
**WeatherPredict** is a bottom-up tropical cyclone forecasting pipeline. It predicts **monthly storm counts** by first classifying storm presence at every 0.25° grid cell per day, then rolling up probabilities into a Storm Potential Index (SPI).
**Domain:** South China Sea / Western Pacific, 1983–2024.

### Tech Stack
- **Data Engineering:** PySpark (rolling features, joins, distributed inference via `mapInPandas`)
- **Micro-level Classifier:** LightGBM (native Python API)
- **Monthly Regression:** scikit-learn (Poisson GridSearchCV, isotonic calibration)
- **Local Training & Processing:** pandas / numpy / scipy
- **Visualization:** Streamlit (`app/app.py`)
- **Environment:** Python 3.11+, Conda env: `pyspark`

### Pipeline Phases
1. **Feature Engineering** (`phase1_features.py`): Rolling averages (7/14/30/90/180 day) + derived features. Persistently cached to eliminate redundant calculation.
2. **Undersampling** (`phase2_sampling.py`): 1:20 positive/negative ratio via fast uniform sampling.
3. **Temporal Split** (`phase3_split.py`): Train ≤2015, Val 2016–2019, Test ≥2020. No random splits to prevent data leakage.
4. **LightGBM Classifier** (`phase4_classifier.py`): Trains on pandas. Outputs grid-level probability: `prob_storm` (ROC-AUC ~0.99).
5. **Monthly Roll-Up** (`phase5_rollup.py`): `mapInPandas` inference + Landfall Spatial Transition Matrix → Country-specific SPIs → Phase 5 Stacked Ensemble (ZINB + Tweedie LightGBM + Ridge) → Country-level landfall predictions.

### How to Run

**1. Setup Environment**
```bash
conda activate pyspark
pip install -r requirements.txt
```

**2. Running the Pipeline**
```bash
# Full pipeline (first run: ~45 min; cached: ~20 min)
python -m models.bottom_up_forecast

# Prepare monthly cache only — no training
python -m models.bottom_up_forecast --prepare

# Train Phase 5 from cache — no Spark needed (~3 min)
python -m models.bottom_up_forecast --phase5
```

**3. Using the Interactive Training Script (Recommended)**
```bash
./models/train.zsh              # Interactive menu
./models/train.zsh bottom_up    # Full pipeline
./models/train.zsh prepare      # Build monthly cache
./models/train.zsh phase5       # Train from cache
```

**4. Dashboard**
```bash
streamlit run app/app.py        # Phase 5 predictions (Actual vs Predicted)
```

**5. Cluster Mode (2-node Standalone)**
See `docs/cluster_setup.md` for full instructions.

### Important Constraints
- **No SynapseML:** Use native `lightgbm` Python API only. SynapseML causes `JavaPackage` errors.
- **No random splits:** Time-based splits only. Future data must never leak into training.
- **Inference requires `mapInPandas`:** Uses a broadcasted pickled model dynamically applied. Do not try `spark.ml` transformers.
- **Two-level caching:** Phase 1 features (`features_checkpoint.parquet`) and Monthly data (`phase5_monthly_cache.parquet`) significantly shortcut training times on reruns.

### Results & Visualizations

#### Phase 4 — LightGBM Grid-Level Storm Classifier

**Training Convergence** — ROC-AUC and PR-AUC over 200 boosting rounds. Validation AUC = 0.9915.

![LightGBM Convergence](docs/chart_lgbm_convergence.png)

**Top 10 Feature Importances** — Sea-level pressure (`slp_avg`) and geographic location (`lon`, `lat`) dominate.

![Feature Importance](docs/chart_feature_importance.png)

#### Phase 5 — Monthly Storm Count Predictions

**Model Comparison** — Ridge (α=10) achieves the lowest Test MAE = 1.03 with minimal overfit.

![Model Comparison](docs/chart_model_comparison.png)

**Annual Totals (Test Set ≥ 2020)** — Predicted vs actual annual storm counts. Annual MAE = 4.05 storms/year.

![Annual Predictions](docs/chart_annual_predictions.png)

**Monthly Detail by Year** — Faceted actual vs predicted for each test year. Overall monthly MAE = 0.25.

![Monthly All Years](docs/chart_monthly_all_years.png)

**2024 Monthly Forecast** — Detailed view of the most recent forecast year with peak season highlighted.

![Monthly 2024](docs/chart_monthly_2024.png)

**Seasonal Climatology** — Model predictions closely follow the 42-year historical pattern (1983–2024).

![Seasonal Pattern](docs/chart_seasonal_pattern.png)

**Error Analysis** — Residual distribution (mean = +0.04) and scatter plot (Pearson r = 0.990).

![Error Distribution](docs/chart_error_distribution.png)

**SPI Correlation (2024)** — Storm Potential Index tracks observed storm count throughout the year.

![SPI Correlation](docs/chart_spi_correlation.png)

**Country-Level Landfall Predictions** — Stacked breakdown by Philippines, Vietnam, Taiwan, China, and Open Sea.

![Country Landfall](docs/chart_country_landfall.png)

---

## Phiên bản Tiếng Việt

### Tổng quan Dự án
**WeatherPredict** là một pipeline dự báo bão nhiệt đới theo phương pháp từ dưới lên (bottom-up). Hệ thống dự đoán **số lượng bão hàng tháng** bằng cách trước tiên phân loại sự hiện diện của bão tại mỗi ô lưới 0.25° mỗi ngày, sau đó tổng hợp các xác suất này thành Chỉ số Tiềm năng Bão (Storm Potential Index - SPI).
**Phạm vi không gian:** Biển Đông / Tây Thái Bình Dương, 1983–2024.

### Công nghệ sử dụng
- **Data Engineering:** PySpark (tính toán đặc trưng dạng trượt - rolling features, join, suy luận phân tán qua `mapInPandas`)
- **Phân loại vi mô (Micro-level):** LightGBM (Native Python API)
- **Hồi quy hàng tháng:** scikit-learn (Poisson GridSearchCV, hiệu chỉnh isotonic)
- **Huấn luyện & Xử lý cục bộ:** pandas / numpy / scipy
- **Trực quan hóa:** Streamlit (`app/app.py`)
- **Môi trường:** Python 3.11+, Conda env: `pyspark`

### Các Giai đoạn của Pipeline
1. **Trích xuất Đặc trưng** (`phase1_features.py`): Tính trung bình trượt (7/14/30/90/180 ngày) + các đặc trưng dẫn xuất. Được lưu cache vĩnh viễn hạn chế tính toán lại dư thừa.
2. **Undersampling** (`phase2_sampling.py`): Giảm mẫu theo tỷ lệ 1:20 (có bão/không bão) bằng phương pháp lấy mẫu đồng đều nhanh.
3. **Chia tập Dữ liệu theo Thời gian** (`phase3_split.py`): Train ≤2015, Val 2016–2019, Test ≥2020. Hoàn toàn không chia tách ngẫu nhiên (chống rò rỉ dữ liệu chiều tương lai).
4. **Mô hình LightGBM** (`phase4_classifier.py`): Huấn luyện trên pandas. Trả về xác suất mức độ ô lưới `prob_storm` (ROC-AUC ~0.99).
5. **Tổng hợp hàng Tháng** (`phase5_rollup.py`): Suy luận phân tán bằng `mapInPandas` + Ma trận Chuyển đổi Không gian Đổ bộ (Landfall Transition Matrix) → SPI theo Quốc gia → Ensemble Mô hình Phase 5 (ZINB + Tweedie LightGBM + Ridge) → Dự báo số lượng bão đổ bộ cấp quốc gia học đa luồng (multi-threaded).

### Hướng dẫn Chạy

**1. Cài đặt Môi trường**
```bash
conda activate pyspark
pip install -r requirements.txt
```

**2. Chạy Pipeline**
```bash
# Chạy toàn bộ pipeline (Lần đầu: ~45 phút; Đã cache: ~20 phút)
python -m models.bottom_up_forecast

# Chỉ tính toán và chuẩn bị bộ cache hàng tháng — không huấn luyện
python -m models.bottom_up_forecast --prepare

# Chạy Phase 5 từ bộ cache — không cần module Spark (~3 phút)
python -m models.bottom_up_forecast --phase5
```

**3. Dùng Script Huấn luyện Tương tác (Khuyên dùng)**
```bash
./models/train.zsh              # Hiện menu tương tác
./models/train.zsh bottom_up    # Chạy toàn bộ pipeline
./models/train.zsh prepare      # Tạo bộ cache hàng tháng
./models/train.zsh phase5       # Huấn luyện Phase 5 trực tiếp từ cache
```

**4. Giao diện Dashboard**
```bash
streamlit run app/app.py        # Dự báo từ Phase 5 (Thực tế vs Dự báo)
```

**5. Chế độ Cluster (Cụm 2 node Standalone)**
Vui lòng tham khảo tệp `docs/cluster_setup.md` để biết thêm chi tiết phương pháp cấu hình mạng linh hoạt.

### Ràng buộc & Lưu ý Quan trọng
- **Không dùng SynapseML:** Chỉ sử dụng native `lightgbm` Python API. Chạy LightGBM bằng SynapseML trong môi trường này gây ra lỗi khởi tạo `JavaPackage`.
- **Không chia dữ liệu ngẫu nhiên (No random splits):** Bắt buộc chỉ chia dữ liệu theo chiều thời gian (Time-based).
- **Suy luận bằng `mapInPandas`:** Thực hiện bằng việc truyền mô hình đã pickle sang các worker (broadcast state). Khác so với truyền thống của Spark ML.
- **Caching 2 lớp:** Đặc trưng Phase 1 (`features_checkpoint.parquet`) và Dữ liệu hàng tháng (`phase5_monthly_cache.parquet`) được lưu trữ độc lập để bỏ qua tính toán phân tán đắt đỏ bằng Spark ở các lần chạy khởi tạo lại.

### Kết quả & Trực quan hóa

#### Phase 4 — Phân loại Bão ở mức Ô lưới (LightGBM)

**Đường cong Hội tụ** — ROC-AUC và PR-AUC qua 200 vòng boosting. Validation AUC = 0.9915.

![LightGBM Convergence](docs/chart_lgbm_convergence.png)

**Top 10 Đặc trưng Quan trọng nhất** — Áp suất mực nước biển (`slp_avg`) và vị trí địa lý (`lon`, `lat`) chiếm ưu thế.

![Feature Importance](docs/chart_feature_importance.png)

#### Phase 5 — Dự báo Số lượng Bão hàng Tháng

**So sánh Mô hình** — Ridge (α=10) đạt Test MAE thấp nhất = 1.03, ít overfit nhất.

![Model Comparison](docs/chart_model_comparison.png)

**Tổng số Bão hàng Năm (Test ≥ 2020)** — So sánh tổng số bão dự báo vs thực tế. MAE hàng năm = 4.05 bão/năm.

![Annual Predictions](docs/chart_annual_predictions.png)

**Chi tiết Dự báo hàng Tháng** — So sánh thực tế vs dự báo cho từng năm test. MAE tổng = 0.25.

![Monthly All Years](docs/chart_monthly_all_years.png)

**Dự báo năm 2024** — Chi tiết dự báo tháng với vùng đỉnh mùa bão được đánh dấu.

![Monthly 2024](docs/chart_monthly_2024.png)

**Chu kỳ Mùa bão** — Dự báo bám sát quy luật khí hậu 42 năm (1983–2024).

![Seasonal Pattern](docs/chart_seasonal_pattern.png)

**Phân tích Sai số** — Phân phối sai số (mean = +0.04) và biểu đồ phân tán (Pearson r = 0.990).

![Error Distribution](docs/chart_error_distribution.png)

**Tương quan SPI (2024)** — Chỉ số Tiềm năng Bão (SPI) phản ánh chính xác đỉnh mùa bão trong năm.

![SPI Correlation](docs/chart_spi_correlation.png)

**Dự báo Đổ bộ theo Quốc gia** — Philippines, Việt Nam, Đài Loan, Trung Quốc và Ngoài khơi.

![Country Landfall](docs/chart_country_landfall.png)
