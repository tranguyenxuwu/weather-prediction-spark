# WeatherPredict: Master Dataset Documentation and ML Recommendations

The [spatio_temporal_join.py](file:///Volumes/ExternalSSD/StudyMaterials/252_SPARK/WeatherPredict/spatio_temporal_join.py) script successfully joined four datasets (ERA5, NOAA SST, ONI, and IBTrACS) into an environment-centric master dataset. This document serves as a data dictionary and best-practice guide for downstream Machine Learning.

## Data Dictionary
The master dataset is stored at `parquet_data/master_dataset.parquet` and is partitioned by `year` and `month`. It contains **~224 million rows** representing daily averages per 0.25° grid cell.

### Temporal and Spatial Coordinates
| Column | Type | Description |
|---|---|---|
| `date` | Date | The daily timestamp for the row |
| `year`, `month` | Int | Derived from date, used for efficient querying/partitioning |
| `lat`, `lon` | Double | Grid cell center coordinates (0.25° resolution, South China Sea region) |

### Environmental Features (The Predictors)
| Column | Type | Description |
|---|---|---|
| `u_wind_avg` | Float | **ERA5**: Daily average zonal (East-West) wind component at 10m (m/s) |
| `v_wind_avg` | Float | **ERA5**: Daily average meridional (North-South) wind component at 10m (m/s) |
| `slp_avg` | Float | **ERA5**: Daily average Mean Sea Level Pressure (Pa) |
| `wind_speed_env_avg`| Float | **ERA5**: Daily average scalar wind speed (m/s) |
| `sst_avg` | Float | **NOAA**: Daily average Sea Surface Temperature (°C). *Note: Null over land masses.* |
| `oni_value` | Float | **CPC**: Oceanic Niño Index for the given month |
| `enso_phase` | Int | **CPC**: ENSO classification (`0` = Neutral, `1` = El Niño, `2` = La Niña) |

### Storm Targets (The Labels)
These columns are derived from IBTrACS. **They are `NULL` for the vast majority of rows.** They are non-null *only* when a storm was present at that specific `lat`/`lon` on that `date`.

| Column | Type | Description |
|---|---|---|
| `SID` | String | Unique storm identifier |
| `NAME` | String | Storm name (can be 'NOT_NAMED') |
| `wind_speed_kmh`| Float | Maximum sustained wind speed of the storm (km/h) |
| `pressure_wmo` | Float | Minimum central pressure of the storm (mb) |

---

## Machine Learning Recommendations

The dataset is extremely rich but highly imbalanced (millions of empty grid cells vs. thousands of storm points). Here are recommendations on how to approach your modeling.

### 1. Handling the Extreme Class Imbalance
Over 99.9% of the dataset rows DO NOT contain a storm (`SID IS NULL`). If you train a model to simply predict "Is there a storm here today?", it will achieve 99.9% accuracy by always predicting "No".

**Recommendations:**
*   **For Classification (Storm Genesis):** Use heavy undersampling of the "No Storm" class. For example, for every 1 true storm point, randomly sample 5-10 environment grids where no storm exists, perhaps restricting the sample to the general cyclone season (June–November) so the model doesn't just learn that "January is safe."
*   **For Regression (Storm Intensity):** Filter the dataset to `WHERE SID IS NOT NULL`. Train the model *only* on the locations where storms already exist, using the environmental features (`sst_avg`, `slp_avg`, `enso_phase`) to predict `wind_speed_kmh` or `pressure_wmo`.

### 2. Feature Engineering

The current features are point-in-time and point-in-space values. Machine Learning models will perform much better if you provide them with spatial and temporal context.

**Recommendations:**
*   **Temporal Aggregations (Lags & Averages):** Storms don't react instantly to today's SST; they respond to long-term oceanic heat content. Use Spark window functions (over time partitioned by grid location) to create trailing average features, such as **1-month, 3-month, and 6-month SST averages**. These long-term averages act as a proxy for total ocean heat potential.
*   **Spatial Aggregations:** A storm draws energy from a wide area, not just a single 0.25° grid point. Consider creating a spatial average feature (e.g., average SST in a 1° radius around the grid point).

### 3. Proof of Concept: Correlation Analysis
Before building complex ML models, run a basic PySpark script to validate the core physical hypotheses:

1.  **SST vs Intensity:** Filter for `SID IS NOT NULL` and calculate the Pearson correlation between `sst_avg` and `wind_speed_kmh`. (Hypothesis: Higher SST = Higher Wind Speed).
2.  **ENSO vs Storm Frequency:** Group by `year, enso_phase` and count the `count(distinct SID)`. (Hypothesis: El Niño / La Niña phases shift the frequency and genesis locations of storms in the South China Sea).

### 4. Splitting the Data

Because weather is a time-series process, **do not use random train/test splits**.
If you randomly split, you might train a model on October 15th's weather and test it on October 14th's weather, causing data leakage.

**Recommendation:**
*   Use a **time-based split**. For example:
    *   **Train:** 1983 - 2015
    *   **Validation:** 2016 - 2019
    *   **Test:** 2020 - 2024 (To simulate exactly how well the model predicts the "unseen future").
