import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from visualization_helper import generate_sst_map

def main():
    base_dir = Path(__file__).parent
    parquet_path = base_dir / "parquet_data" / "noaa_sst" / "year=1981" / "month=1" / "*.parquet"
    
    print(f"Loading data from {parquet_path}...")
    try:
        query = f"SELECT time, lat, lon, sst FROM read_parquet('{str(parquet_path)}')"
        con = duckdb.connect(database=':memory:')
        df = con.execute(query).fetchdf()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    if df.empty:
        print("No data found.")
        return

    print("Data loaded. Filtering for first timestamp...")
    timestamps = sorted(df['time'].unique())
    if not timestamps:
        print("No timestamps found.")
        return
        
    first_time = timestamps[0]
    print(f"Selected time: {first_time}")
    
    subset = df[df['time'] == first_time].copy()
    subset = subset.dropna(subset=['lat', 'lon', 'sst'])
    
    print(f"Generating map for {len(subset)} points...")
    try:
        fig = generate_sst_map(subset)
        output_path = base_dir / "verification_map.png"
        fig.savefig(output_path)
        print(f"Map saved to {output_path}")
    except Exception as e:
        print(f"Failed to generate map: {e}")

if __name__ == "__main__":
    main()
