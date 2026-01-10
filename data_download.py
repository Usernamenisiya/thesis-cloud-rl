import requests
import os
from pathlib import Path
import numpy as np
import rasterio
import zarr
from scipy.ndimage import zoom

# Create data directory
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

def download_file(url, dest_path):
    """Download a file from URL to dest_path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {dest_path}")

def download_sentinel2_sample():
    """Create a dummy Sentinel-2 image with 12 bands."""
    # For demo, create a dummy .tif with 12 bands, shape 256x256
    import numpy as np
    height, width = 256, 256
    bands = 12
    # Simulate reflectance values (0-1)
    image = np.random.rand(bands, height, width).astype(np.float32)
    
    meta = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': width,
        'height': height,
        'count': bands,
        'crs': 'EPSG:4326',  # Dummy CRS
        'transform': rasterio.transform.from_bounds(0, 0, 1, 1, width, height)
    }
    
    sentinel_path = data_dir / "sentinel2_image.tif"
    with rasterio.open(sentinel_path, 'w', **meta) as dst:
        for i in range(bands):
            dst.write(image[i], i+1)
    
    print("Dummy Sentinel-2 image created.")
    return sentinel_path

import zarr
import shutil

def copy_and_unzip_zarr(zip_path):
    """Copy the downloaded zip to data/ and unzip."""
    dest_zip = data_dir / "sentinel2.zarr.zip"
    shutil.copy(zip_path, dest_zip)
    
    # Unzip
    import zipfile
    with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
        zip_ref.extractall(data_dir / "sentinel2_zarr")
    
    return data_dir / "sentinel2_zarr"

def load_and_stack_bands(zarr_dir):
    """Load bands from ZARR and stack into .tif."""
    # Bands: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
    # B02-B07, B8A, B11, B12 from r20m; B08 from r10m (resample to 20m)
    bands_20m = ['b02', 'b03', 'b04', 'b05', 'b06', 'b07', 'b8a', 'b11', 'b12']
    bands_10m = ['b08']
    
    bands = []
    
    # Load 20m bands
    for b in bands_20m:
        zarr_path = zarr_dir / "measurements" / "reflectance" / "r20m" / b
        if zarr_path.exists():
            arr = zarr.open(str(zarr_path), mode='r')[:]
            bands.append(arr)
        else:
            print(f"Band {b} not found.")
            return None
    
    # Load B08 from 10m and resample to 20m
    zarr_path_08 = zarr_dir / "measurements" / "reflectance" / "r10m" / "b08"
    if zarr_path_08.exists():
        arr_08 = zarr.open(str(zarr_path_08), mode='r')[:]
        # Resample from 10m to 20m: assume shape is 2x, use zoom with order=0 (nearest)
        arr_08_resampled = zoom(arr_08, 0.5, order=0)
        bands.insert(6, arr_08_resampled)  # Insert at position 6 (after B07)
    else:
        print("Band b08 not found.")
        return None
    
    # Now bands order: b02, b03, b04, b05, b06, b07, b08, b8a, b11, b12
    
    # Stack to (bands, H, W)
    stacked = np.stack(bands, axis=0)
    
    # Get coordinate info from ZARR
    x_coords = zarr.open(str(zarr_dir / "measurements" / "reflectance" / "r20m" / "x"), mode='r')[:]
    y_coords = zarr.open(str(zarr_dir / "measurements" / "reflectance" / "r20m" / "y"), mode='r')[:]
    
    # Calculate pixel size and transform
    pixel_size = 20.0  # 20m resolution
    transform = rasterio.transform.from_bounds(
        x_coords.min(), y_coords.min(), 
        x_coords.max() + pixel_size, y_coords.max() + pixel_size,
        stacked.shape[2], stacked.shape[1]
    )
    
    meta = {
        'driver': 'GTiff',
        'dtype': stacked.dtype,
        'width': stacked.shape[2],
        'height': stacked.shape[1],
        'count': len(bands),
        'crs': 'EPSG:32725',  # UTM Zone 25S
        'transform': transform
    }
    
    tif_path = data_dir / "sentinel2_image.tif"
    with rasterio.open(tif_path, 'w', **meta) as dst:
        for i in range(len(bands)):
            dst.write(stacked[i], i+1)
    
    return tif_path

if __name__ == "__main__":
    # Path to your downloaded file
    downloaded_zip = r"C:\Users\123\Downloads\S2B_MSIL2A_20251218T125029_N0511_R095_T25SGB_20251218T132233.zarr.zip"
    
    try:
        zarr_dir = copy_and_unzip_zarr(downloaded_zip)
        tif_path = load_and_stack_bands(zarr_dir)
        if tif_path:
            # download_ground_truth_sample()
            print("Data preparation complete.")
        else:
            print("Failed to stack bands.")
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure the ZARR structure matches expectations.")