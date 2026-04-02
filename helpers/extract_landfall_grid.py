import json
import pandas as pd
import numpy as np
from shapely.geometry import shape, Point
import os

print("── Phase 1: Loading Country Polygons ──")
with open("world-countries.json", "r") as f:
    geojson = json.load(f)

countries = {}
target_names = ["Vietnam", "Philippines", "China", "Taiwan", "Japan"]
name_mapping = {
    "Vietnam": "VN",
    "Philippines": "PH",
    "China": "CN",
    "Taiwan": "TW",
    "Japan": "JP"
}

for feature in geojson["features"]:
    props = feature["properties"]
    name = props.get("name", props.get("ADMIN", ""))
    for t in target_names:
        if t.lower() in name.lower():
            if name_mapping[t] not in countries:
                # Store the Shapely polygon/multipolygon
                countries[name_mapping[t]] = shape(feature["geometry"])

print(f"Found country polygons: {list(countries.keys())}")

print("── Phase 2: Loading IBTrACS Tracks ──")
df = pd.read_parquet("parquet_data/ibtracs_clean.parquet")
df = df.sort_values(["SID", "timestamp"])

print("── Phase 3: Determining Historical Landfalls ──")
storm_landfalls = {}
sids = df["SID"].unique()

for i, sid in enumerate(sids):
    if i % 500 == 0:
        print(f"  Processed {i}/{len(sids)} storms...")
        
    group = df[df["SID"] == sid]
    lats = group["lat"].values
    lons = group["lon"].values
    
    landfall = "None"
    
    # Trace the storm chronologically. The first target it hits becomes its ultimate landfall.
    for lat, lon in zip(lats, lons):
        pt = Point(lon, lat)  # shapely uses (x, y) -> (lon, lat)
        hit = False
        for ccode, poly in countries.items():
            # 0.5 degree buffer ~ 50km proximity to coastline
            if poly.distance(pt) < 0.5: 
                landfall = ccode
                hit = True
                break
        if hit:
            break
            
    storm_landfalls[sid] = landfall

# Global statistics
counts = pd.Series(storm_landfalls).value_counts()
print(f"\nTotal Landfall Distribution (historical 1980-2024):")
print(counts)

print("\n── Phase 4: Building Spatial Transition Matrix ──")
# Map targets to points
df["landfall_target"] = df["SID"].map(storm_landfalls)

# Snap to grid
df["lat_grid"] = (df["lat"] * 4).round() / 4
df["lon_grid"] = (df["lon"] * 4).round() / 4

# Drop duplicate visits to the same grid cell by the same storm (prevents slow-moving storms from over-weighting)
unique_cell_visits = df[["SID", "lat_grid", "lon_grid", "landfall_target"]].drop_duplicates()

# Cross-tabulate cells by ultimate landfall destination
cell_stats = unique_cell_visits.groupby(["lat_grid", "lon_grid", "landfall_target"]).size().unstack(fill_value=0)

target_cols = ["CN", "JP", "None", "PH", "TW", "VN"]
for col in target_cols:
    if col not in cell_stats.columns:
        cell_stats[col] = 0

cell_stats = cell_stats[target_cols]

row_sums = cell_stats.sum(axis=1)

# Bayesian Smoothing (Laplace smoothing with basin prior)
# For cells with very few historical storms, we regress toward the overall basin average.
basin_avg = unique_cell_visits["landfall_target"].value_counts(normalize=True).reindex(target_cols).fillna(0).values
alpha = 5  # "pseudocount" smoothing strength

smoothed_probs = (cell_stats + alpha * basin_avg).div(row_sums + alpha, axis=0)

# Prefix columns
smoothed_probs.columns = [f"prob_{c}" for c in smoothed_probs.columns]
smoothed_probs = smoothed_probs.reset_index()

out_path = "parquet_data/landfall_transition_grid.parquet"
smoothed_probs.to_parquet(out_path, index=False)
print(f"✅ Successfully saved transition matrix to {out_path} ({len(smoothed_probs)} grid cells mapped).")

# Save storm landfalls mapping to be joined as ground truth in phase 5
storm_targets_df = pd.DataFrame(list(storm_landfalls.items()), columns=["SID", "landfall_target"])
targets_path = "parquet_data/storm_landfalls.parquet"
storm_targets_df.to_parquet(targets_path, index=False)
print(f"✅ Successfully saved storm target mapping to {targets_path} ({len(storm_targets_df)} storms).")
