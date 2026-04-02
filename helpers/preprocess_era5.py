import xarray as xr
import numpy as np
import os
import pandas as pd

INPUT_DIR = "SPARK__DATA/era5_data"
OUTPUT_DIR = "SPARK__DATA/processed/era5"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.nc')])

print(f"📁 Tìm thấy {len(files)} file ERA5 NetCDF")

for i, f in enumerate(files, 1):
    print(f"🔄 [{i}/{len(files)}] Đang xử lý {f}...")

    # 1. Mở file
    ds = xr.open_dataset(os.path.join(INPUT_DIR, f))

    # 2. Đổi tên cho dễ hiểu
    rename_map = {}
    if 'u10' in ds.data_vars:
        rename_map['u10'] = 'u_wind'
    if 'v10' in ds.data_vars:
        rename_map['v10'] = 'v_wind'
    if 'msl' in ds.data_vars:
        rename_map['msl'] = 'slp'  # SLP: Sea Level Pressure
    ds = ds.rename(rename_map)

    # 3. Tính tốc độ gió tổng hợp (Magnitude)
    # Công thức: sqrt(u^2 + v^2)
    ds['wind_speed_env'] = np.sqrt(ds['u_wind']**2 + ds['v_wind']**2)

    # 4. Chuyển sang Pandas DataFrame
    # ERA5 vùng cắt nhỏ (~121x121x1460), có thể load vào RAM
    df = ds.to_dataframe().reset_index()

    # 5. Chuẩn hóa cột thời gian (bỏ timezone nếu có)
    if 'valid_time' in df.columns:
        df['valid_time'] = pd.to_datetime(df['valid_time'])

    # 6. Xóa cột thừa (number, expver)
    cols_to_drop = ['number', 'expver']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 7. Lưu Parquet
    out_name = f.replace('.nc', '.parquet')
    df.to_parquet(os.path.join(OUTPUT_DIR, out_name), index=False)

    ds.close()

print(f"\n✅ Đã xử lý xong {len(files)} file ERA5!")
