#!/usr/bin/env python3
"""
CloudSEN12 Data Loader for RL Training.
Converts CloudSEN12 patches into the format expected by existing training code.
"""
import os
import numpy as np
import rasterio
from pathlib import Path

def load_cloudsen12_patches(cloudsen_dir, max_patches=20):
    """
    Load CloudSEN12 patches and combine into training dataset.
    
    Args:
        cloudsen_dir: Path to CloudSEN12 subset folder
        max_patches: Maximum number of patches to use
    
    Returns:
        images: List of Sentinel-2 images (H, W, bands)
        masks: List of ground truth masks (H, W)
    """
    print("=" * 60)
    print("📦 Loading CloudSEN12 Data")
    print("=" * 60)
    
    cloudsen_path = Path(cloudsen_dir)
    patch_dirs = sorted([d for d in cloudsen_path.iterdir() if d.is_dir()])[:max_patches]
    
    print(f"\n✅ Found {len(patch_dirs)} patches to load")
    
    images = []
    masks = []
    
    for idx, patch_dir in enumerate(patch_dirs):
        print(f"  Loading patch {idx+1}/{len(patch_dirs)}: {patch_dir.name}", end='\r')
        
        s2l1c_path = patch_dir / "s2l1c.tif"
        target_path = patch_dir / "target.tif"
        
        if not s2l1c_path.exists() or not target_path.exists():
            print(f"\n⚠️  Skipping {patch_dir.name} - missing files")
            continue
        
        try:
            # Load Sentinel-2 image
            with rasterio.open(s2l1c_path) as src:
                image = src.read()  # (bands, H, W)
                image = np.transpose(image, (1, 2, 0))  # (H, W, bands)
                
                # Select 10 standard bands (B2-B8A, B11-B12)
                # CloudSEN12 has 13 bands: B1-B12 + B8A
                # We exclude B1 (coastal aerosol), B9 (water vapor), B10 (cirrus)
                # Standard order: [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]
                # CloudSEN12 order: [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12]
                # Indices to keep: [1, 2, 3, 4, 5, 6, 7, 8, 11, 12] (0-indexed)
                band_indices = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
                image = image[:, :, band_indices]  # (H, W, 10)
                
                # Normalize to 0-1 range (assuming 16-bit data)
                image = image.astype(np.float32) / 10000.0
                image = np.clip(image, 0, 1)
            
            # Load ground truth mask
            with rasterio.open(target_path) as src:
                mask = src.read(1)  # (H, W)
                # CloudSEN12 uses: 0=clear, 1=cloud shadow, 2=cloud, 3=semi-transparent, 4=invalid
                # Convert to binary: cloud (2,3) = 1, others = 0
                mask = np.isin(mask, [2, 3]).astype(np.uint8)
            
            images.append(image)
            masks.append(mask)
            
        except Exception as e:
            print(f"\n❌ Error loading {patch_dir.name}: {e}")
            continue
    
    print(f"\n\n✅ Successfully loaded {len(images)} patches")
    
    if len(images) > 0:
        print(f"📊 Image shape: {images[0].shape}")
        print(f"📊 Mask shape: {masks[0].shape}")
        print(f"📊 Image bands: {images[0].shape[2]}")
        print(f"📊 Cloud coverage: {np.mean([m.mean() for m in masks])*100:.1f}%")
    
    return images, masks

def save_as_geotiff(image, mask, output_dir, patch_name):
    """
    Save a single patch as GeoTIFF files compatible with existing training code.
    
    Args:
        image: Sentinel-2 image (H, W, bands)
        mask: Ground truth mask (H, W)
        output_dir: Output directory
        patch_name: Name for this patch
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save image
    image_path = output_path / f"{patch_name}_image.tif"
    image_transposed = np.transpose(image, (2, 0, 1))  # (bands, H, W)
    
    with rasterio.open(
        image_path, 'w',
        driver='GTiff',
        height=image.shape[0],
        width=image.shape[1],
        count=image.shape[2],
        dtype=image.dtype
    ) as dst:
        dst.write(image_transposed)
    
    # Save mask
    mask_path = output_path / f"{patch_name}_mask.tif"
    
    with rasterio.open(
        mask_path, 'w',
        driver='GTiff',
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=mask.dtype
    ) as dst:
        dst.write(mask, 1)
    
    return image_path, mask_path

def prepare_cloudsen12_for_training(cloudsen_dir, output_dir="data/cloudsen12_processed", num_patches=5):
    """
    Prepare CloudSEN12 data for RL training.
    Creates individual patch files compatible with existing training pipeline.
    
    Args:
        cloudsen_dir: Path to CloudSEN12 subset folder
        output_dir: Where to save processed files
        num_patches: Number of patches to process
    """
    print("\n" + "=" * 60)
    print("🔧 Preparing CloudSEN12 for Training")
    print("=" * 60)
    
    images, masks = load_cloudsen12_patches(cloudsen_dir, max_patches=num_patches)
    
    if len(images) == 0:
        print("❌ No patches loaded. Exiting.")
        return
    
    print(f"\n💾 Saving {len(images)} patches to {output_dir}")
    
    for idx, (image, mask) in enumerate(zip(images, masks)):
        patch_name = f"patch_{idx:03d}"
        img_path, mask_path = save_as_geotiff(image, mask, output_dir, patch_name)
        print(f"  Saved: {patch_name}")
    
    print(f"\n✅ Data preparation complete!")
    print(f"📂 Files saved to: {output_dir}")
    print(f"\n🎯 Ready for training with real ground truth!")

if __name__ == "__main__":
    # For Colab usage
    cloudsen_dir = "/content/drive/MyDrive/Colab_Data/cloudsen12_subset"
    output_dir = "data/cloudsen12_processed"
    
    prepare_cloudsen12_for_training(
        cloudsen_dir=cloudsen_dir,
        output_dir=output_dir,
        num_patches=100  # Process all 100 patches
    )
