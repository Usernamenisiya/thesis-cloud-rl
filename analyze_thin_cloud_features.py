"""
Analyze if thin clouds actually have distinguishable features in CloudSEN12 data.

This will help us understand:
1. Do thin vs thick clouds have different spectral signatures?
2. Is blue/red ratio actually useful for thin cloud detection?
3. What features best separate thin from thick clouds?
"""

import numpy as np
import rasterio
import glob
from cnn_inference import load_sentinel2_image
import matplotlib.pyplot as plt

def analyze_cloud_features(image_files, mask_files, num_samples=10):
    """Analyze spectral features of thin vs thick clouds."""
    
    thin_blue_red = []
    thick_blue_red = []
    thin_reflectance = []
    thick_reflectance = []
    thin_ndvi = []
    thick_ndvi = []
    
    print(f"Analyzing {num_samples} patches...")
    
    for img_path, mask_path in zip(image_files[:num_samples], mask_files[:num_samples]):
        # Load data
        image = load_sentinel2_image(img_path)
        with rasterio.open(mask_path) as src:
            ground_truth = src.read(1)
        
        # Get cloud mask
        cloud_mask = ground_truth > 0
        if cloud_mask.sum() == 0:
            continue
        
        # Extract bands
        if image.shape[2] >= 8:
            blue = image[:, :, 1].astype(np.float32)
            green = image[:, :, 2].astype(np.float32)
            red = image[:, :, 3].astype(np.float32)
            nir = image[:, :, 7].astype(np.float32)
            
            # Calculate features
            blue_red_ratio = np.where(red > 100, blue / (red + 1e-6), 0)
            reflectance = (blue + green + red + nir) / 4.0
            ndvi = np.where(nir + red > 0, (nir - red) / (nir + red + 1e-6), 0)
            
            # Classify based on reflectance percentile
            cloud_reflectance = reflectance[cloud_mask]
            thin_threshold = np.percentile(cloud_reflectance, 70)
            
            thin_mask = np.logical_and(cloud_mask, reflectance < thin_threshold)
            thick_mask = np.logical_and(cloud_mask, reflectance >= thin_threshold)
            
            # Collect features
            if thin_mask.sum() > 0:
                thin_blue_red.extend(blue_red_ratio[thin_mask].flatten())
                thin_reflectance.extend(reflectance[thin_mask].flatten())
                thin_ndvi.extend(ndvi[thin_mask].flatten())
            
            if thick_mask.sum() > 0:
                thick_blue_red.extend(blue_red_ratio[thick_mask].flatten())
                thick_reflectance.extend(reflectance[thick_mask].flatten())
                thick_ndvi.extend(ndvi[thick_mask].flatten())
    
    # Convert to arrays
    thin_blue_red = np.array(thin_blue_red)
    thick_blue_red = np.array(thick_blue_red)
    thin_reflectance = np.array(thin_reflectance)
    thick_reflectance = np.array(thick_reflectance)
    thin_ndvi = np.array(thin_ndvi)
    thick_ndvi = np.array(thick_ndvi)
    
    # Print statistics
    print("\n" + "="*80)
    print("THIN CLOUD FEATURES")
    print("="*80)
    print(f"Blue/Red Ratio:  Mean={thin_blue_red.mean():.3f}, Std={thin_blue_red.std():.3f}")
    print(f"Reflectance:     Mean={thin_reflectance.mean():.1f}, Std={thin_reflectance.std():.1f}")
    print(f"NDVI:            Mean={thin_ndvi.mean():.3f}, Std={thin_ndvi.std():.3f}")
    print(f"Total pixels:    {len(thin_reflectance):,}")
    
    print("\n" + "="*80)
    print("THICK CLOUD FEATURES")
    print("="*80)
    print(f"Blue/Red Ratio:  Mean={thick_blue_red.mean():.3f}, Std={thick_blue_red.std():.3f}")
    print(f"Reflectance:     Mean={thick_reflectance.mean():.1f}, Std={thick_reflectance.std():.1f}")
    print(f"NDVI:            Mean={thick_ndvi.mean():.3f}, Std={thick_ndvi.std():.3f}")
    print(f"Total pixels:    {len(thick_reflectance):,}")
    
    print("\n" + "="*80)
    print("SEPARABILITY ANALYSIS")
    print("="*80)
    
    # Check if features are actually different
    blue_red_diff = abs(thin_blue_red.mean() - thick_blue_red.mean())
    reflectance_diff = abs(thin_reflectance.mean() - thick_reflectance.mean())
    ndvi_diff = abs(thin_ndvi.mean() - thick_ndvi.mean())
    
    print(f"Blue/Red Difference:  {blue_red_diff:.3f} ({'GOOD' if blue_red_diff > 0.1 else 'POOR'})")
    print(f"Reflectance Difference: {reflectance_diff:.1f} ({'GOOD' if reflectance_diff > 500 else 'POOR'})")
    print(f"NDVI Difference:      {ndvi_diff:.3f} ({'GOOD' if ndvi_diff > 0.1 else 'POOR'})")
    
    # Check overlap
    thin_blue_red_range = (thin_blue_red.min(), thin_blue_red.max())
    thick_blue_red_range = (thick_blue_red.min(), thick_blue_red.max())
    
    print(f"\nBlue/Red Ranges:")
    print(f"  Thin:  {thin_blue_red_range[0]:.3f} - {thin_blue_red_range[1]:.3f}")
    print(f"  Thick: {thick_blue_red_range[0]:.3f} - {thick_blue_red_range[1]:.3f}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if blue_red_diff < 0.05:
        print("⚠️  Blue/Red ratio does NOT distinguish thin vs thick clouds well")
        print("    Consider removing this feature or using different spectral bands")
    else:
        print("✅ Blue/Red ratio is useful for thin cloud detection")
    
    if reflectance_diff < 300:
        print("⚠️  Reflectance difference is small - classification might be arbitrary")
    else:
        print("✅ Reflectance-based classification is valid")
    
    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Blue/Red ratio
    axes[0].hist(thin_blue_red, bins=50, alpha=0.5, label='Thin', density=True)
    axes[0].hist(thick_blue_red, bins=50, alpha=0.5, label='Thick', density=True)
    axes[0].set_xlabel('Blue/Red Ratio')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Blue/Red Ratio Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reflectance
    axes[1].hist(thin_reflectance, bins=50, alpha=0.5, label='Thin', density=True)
    axes[1].hist(thick_reflectance, bins=50, alpha=0.5, label='Thick', density=True)
    axes[1].set_xlabel('Reflectance')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Reflectance Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # NDVI
    axes[2].hist(thin_ndvi, bins=50, alpha=0.5, label='Thin', density=True)
    axes[2].hist(thick_ndvi, bins=50, alpha=0.5, label='Thick', density=True)
    axes[2].set_xlabel('NDVI')
    axes[2].set_ylabel('Density')
    axes[2].set_title('NDVI Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/thin_cloud_feature_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: results/thin_cloud_feature_analysis.png")
    plt.close()
    
    return {
        'thin_blue_red': thin_blue_red,
        'thick_blue_red': thick_blue_red,
        'thin_reflectance': thin_reflectance,
        'thick_reflectance': thick_reflectance,
        'separability': {
            'blue_red_diff': blue_red_diff,
            'reflectance_diff': reflectance_diff,
            'ndvi_diff': ndvi_diff
        }
    }


if __name__ == "__main__":
    # Load file lists
    processed_dir = 'data/cloudsen12_processed'
    image_files = sorted(glob.glob(f'{processed_dir}/*_image.tif'))
    mask_files = sorted(glob.glob(f'{processed_dir}/*_mask.tif'))
    
    print(f"Found {len(image_files)} patches")
    
    # Analyze features
    results = analyze_cloud_features(image_files, mask_files, num_samples=20)
    
    print("\n✅ Analysis complete!")
