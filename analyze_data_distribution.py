#!/usr/bin/env python3
"""
Analyze cloud/clear patch distribution in the dataset.
"""
import numpy as np
import rasterio

def analyze_patch_distribution(ground_truth_path, patch_size=64):
    """Analyze distribution of cloud vs clear patches."""
    
    # Load ground truth
    with rasterio.open(ground_truth_path) as src:
        ground_truth = src.read(1).astype(np.uint8)
    
    H, W = ground_truth.shape
    print(f"Image shape: {H}x{W}")
    print(f"Patch size: {patch_size}x{patch_size}")
    
    # Count patches by type
    total_patches = 0
    clear_patches = 0  # 0 cloud pixels
    few_clouds = 0     # < 30% cloud pixels
    many_clouds = 0    # >= 30% cloud pixels
    
    cloud_pixel_percentages = []
    
    # Iterate through all patches
    for i in range(0, H - patch_size, patch_size):
        for j in range(0, W - patch_size, patch_size):
            patch = ground_truth[i:i+patch_size, j:j+patch_size]
            cloud_pixels = np.sum(patch > 0)
            total_pixels = patch.size
            cloud_percentage = (cloud_pixels / total_pixels) * 100
            
            cloud_pixel_percentages.append(cloud_percentage)
            total_patches += 1
            
            if cloud_pixels == 0:
                clear_patches += 1
            elif cloud_pixels < total_pixels * 0.3:
                few_clouds += 1
            else:
                many_clouds += 1
    
    # Calculate statistics
    print(f"\n{'='*60}")
    print("📊 PATCH DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    print(f"Total patches: {total_patches}")
    print(f"\nPatch Categories:")
    print(f"  Clear sky (0% clouds):     {clear_patches:5d} ({clear_patches/total_patches*100:5.1f}%)")
    print(f"  Few clouds (<30%):         {few_clouds:5d} ({few_clouds/total_patches*100:5.1f}%)")
    print(f"  Many clouds (≥30%):        {many_clouds:5d} ({many_clouds/total_patches*100:5.1f}%)")
    
    print(f"\n{'='*60}")
    print("📈 CLOUD PERCENTAGE STATISTICS")
    print(f"{'='*60}")
    cloud_pixel_percentages = np.array(cloud_pixel_percentages)
    print(f"Mean cloud coverage per patch: {np.mean(cloud_pixel_percentages):.2f}%")
    print(f"Median cloud coverage:         {np.median(cloud_pixel_percentages):.2f}%")
    print(f"Std deviation:                 {np.std(cloud_pixel_percentages):.2f}%")
    print(f"Min coverage:                  {np.min(cloud_pixel_percentages):.2f}%")
    print(f"Max coverage:                  {np.max(cloud_pixel_percentages):.2f}%")
    
    # Calculate class imbalance
    cloudy_patches = few_clouds + many_clouds
    imbalance_ratio = clear_patches / cloudy_patches if cloudy_patches > 0 else float('inf')
    
    print(f"\n{'='*60}")
    print("⚖️  CLASS IMBALANCE")
    print(f"{'='*60}")
    print(f"Clear patches:  {clear_patches}")
    print(f"Cloudy patches: {cloudy_patches}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1 (clear:cloudy)")
    
    if imbalance_ratio > 2:
        print(f"\n⚠️  HIGH CLASS IMBALANCE DETECTED!")
        print(f"    Clear patches outnumber cloudy by {imbalance_ratio:.1f}x")
        print(f"    This explains why the agent learned to predict 'clear' always.")
    
    return {
        'total_patches': total_patches,
        'clear_patches': clear_patches,
        'few_clouds': few_clouds,
        'many_clouds': many_clouds,
        'mean_coverage': np.mean(cloud_pixel_percentages),
        'imbalance_ratio': imbalance_ratio
    }

if __name__ == "__main__":
    stats = analyze_patch_distribution("data/ground_truth.tif", patch_size=64)
