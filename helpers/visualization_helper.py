import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_sst_map(df):
    """
    Generates a static matplotlib figure for SST data using contour plots.
    
    Args:
        df: Pandas DataFrame with columns 'lat', 'lon', 'sst'.
        
    Returns:
        fig: Matplotlib Figure object.
    """
    # Setup Figure
    # Use a wide aspect ratio closer to the map
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set background color to represent "land" (since we only plot ocean data)
    # Using a tan/earthy color similar to the reference
    ax.set_facecolor('#d2b48c') 
    
    # Extract data
    lats = df['lat'].values
    lons = df['lon'].values
    sst = df['sst'].values
    
    # Normalize Longitudes to -180 to 180 range if they are 0-360
    # The reference image seems to have 0 at the left or center? 
    # Usually standard maps are -180 to 180.
    # Our data loader (seen in app.py) does ((lon + 180) % 360) - 180
    # Let's simple plot as is, assuming the data is clean or we clean it here.
    
    # Ensure -180 to 180 for standard view
    lons = np.array([((x + 180) % 360) - 180 for x in lons])

    # Plotting
    # tricontourf is robust for scattered data (lat/lon pairs)
    # levels=20 gives us distinct "zones" or bands
    # cmap='nipy_spectral' is often used for this kind of rainbow/banded look
    
    # Clean nan values just in case
    mask = ~np.isnan(sst)
    lats = lats[mask]
    lons = lons[mask]
    sst = sst[mask]
    
    if len(sst) == 0:
        return fig

    # Use tricontourf
    levels = np.linspace(np.min(sst), np.max(sst), 21)
    contour = ax.tricontourf(lons, lats, sst, levels=levels, cmap='nipy_spectral', extend='both')
    
    # Remove axes for a cleaner "image only" look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add Land Overlay
    try:
        add_land_overlay(ax)
    except Exception as e:
        print(f"Warning: Could not add land overlay: {e}")
    
    # Optionally add colorbar? 
    # The prompt says "browser only need to show the image", 
    # usually implies a clean map, but a colorbar is helpful.
    # Let's add a thin one at the bottom or right.
    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.02, aspect=30)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Temperature (°C)', rotation=270, labelpad=15)
    
    # Set limits to world map
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    
    # Tight layout to remove whitespace
    plt.tight_layout()
    
    return fig

def add_land_overlay(ax):
    """
    Downloads and plots a world chart on the given axes.
    """
    import json
    import urllib.request
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import os
    
    # URL for world countries GeoJSON (stable source from Folium examples)
    url = "https://raw.githubusercontent.com/python-visualization/folium/main/examples/data/world-countries.json"
    cache_path = "world-countries.json"
    
    if not os.path.exists(cache_path):
        try:
            print(f"Downloading world map from {url}...")
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            # Fallback or strict fail? Let's just fail this part but keep the map
            raise RuntimeError(f"Failed to download map: {e}")
    else:
        with open(cache_path, 'r') as f:
            data = json.load(f)
            
    patches = []
    
    for feature in data['features']:
        geometry = feature['geometry']
        geo_type = geometry['type']
        coordinates = geometry['coordinates']
        
        if geo_type == 'Polygon':
            # List of rings, first is exterior
            for ring in coordinates:
                # ring is list of [lon, lat]
                # Matplotlib Polygon expects (N, 2)
                poly = Polygon(ring, closed=True)
                patches.append(poly)
        elif geo_type == 'MultiPolygon':
            for polygon in coordinates:
                for ring in polygon:
                    poly = Polygon(ring, closed=True)
                    patches.append(poly)
                    
    # Create a PatchCollection
    # Facecolor: Tan (#d2b48c), Edgecolor: Grey (#555555) for borders
    p = PatchCollection(patches, facecolor='#d2b48c', edgecolor='#555555', linewidth=0.5, zorder=2)
    ax.add_collection(p)

