"""
Download CloudSEN12 dataset to local machine using official tacoreader method.

This script downloads 1000 patches (~25GB) to your local data folder.
Uses cloud-optimized streaming - no need to download entire dataset.

Run from: Thesis Final directory
Output: data/cloudsen12_1000patches/
"""

import pathlib
import numpy as np

def download_cloudsen12(num_patches=1000, output_dir="data/cloudsen12_1000patches"):
    """
    Download CloudSEN12 patches using official tacoreader library.
    
    Args:
        num_patches: Number of patches to download (default 1000)
        output_dir: Output directory for GeoTIFF files
    """
    print("=" * 80)
    print("📦 CloudSEN12 Dataset Downloader (Local)")
    print("=" * 80)
    print(f"\nTarget: {num_patches} patches (~{num_patches * 25 / 1000:.1f} GB)")
    print(f"Output: {output_dir}")
    print("\n" + "=" * 80)
    
    # Install tacoreader
    print("\n📥 Installing tacoreader...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tacoreader", "rasterio"])
    
    import tacoreader
    import rasterio as rio
    
    print(f"✅ tacoreader version: {tacoreader.__version__}")
    
    # Load dataset (streaming mode - doesn't download everything)
    print("\n🌐 Loading CloudSEN12 dataset (streaming)...")
    print("Note: This loads metadata only, actual data downloaded on-demand")
    
    dataset = tacoreader.load([
        "cloudsen12-l1c.0000.part.taco",
        "cloudsen12-l1c.0001.part.taco",
        "cloudsen12-l1c.0002.part.taco",
        "cloudsen12-l1c.0003.part.taco"
    ])
    
    total_available = len(dataset)
    print(f"✅ Dataset loaded: {total_available:,} patches available")
    
    if num_patches > total_available:
        print(f"⚠️  Requested {num_patches} patches but only {total_available} available")
        print(f"   Downloading all {total_available} patches instead")
        num_patches = total_available
    
    # Create output folder
    out_folder = pathlib.Path(output_dir)
    out_folder.mkdir(exist_ok=True, parents=True)
    
    print(f"\n📥 Downloading {num_patches} patches...")
    print("This will take a while (~10-30 minutes depending on internet speed)")
    print()
    
    # Download patches
    downloaded = 0
    failed = 0
    
    for index in range(num_patches):
        try:
            if (index + 1) % 50 == 0:
                print(f"  Progress: {index + 1}/{num_patches} patches ({(index + 1) / num_patches * 100:.1f}%)")
            
            # Read sample
            image = dataset.read(index)
            
            # Create output paths
            patch_id = dataset.iloc[index]["tortilla:id"]
            s2l1c_output = out_folder / patch_id / "s2l1c.tif"
            target_output = out_folder / patch_id / "target.tif"
            target_output.parent.mkdir(exist_ok=True, parents=True)
            
            # Read and write S2 image and target mask
            with rio.open(image.read(0)) as src, rio.open(image.read(1)) as dst:
                # Write S2 L1C image
                with rio.open(s2l1c_output, "w", **src.profile) as src_out:
                    src_out.write(src.read())
                
                # Write target cloud mask
                with rio.open(target_output, "w", **dst.profile) as dst_out:
                    dst_out.write(dst.read())
            
            downloaded += 1
            
        except Exception as e:
            print(f"  ⚠️  Failed to download patch {index}: {e}")
            failed += 1
            continue
    
    print("\n" + "=" * 80)
    print("✅ Download Complete!")
    print("=" * 80)
    print(f"Successfully downloaded: {downloaded} patches")
    if failed > 0:
        print(f"Failed: {failed} patches")
    print(f"\nOutput directory: {out_folder.absolute()}")
    print(f"\nFolder structure:")
    print(f"  {output_dir}/")
    print(f"    ROI_XXXX_<timestamp>_<tile>/")
    print(f"      s2l1c.tif   (Sentinel-2 L1C image)")
    print(f"      target.tif  (Cloud mask labels)")
    print("\n" + "=" * 80)
    
    return downloaded, failed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download CloudSEN12 dataset")
    parser.add_argument("--num-patches", type=int, default=1000,
                        help="Number of patches to download (default: 1000)")
    parser.add_argument("--output-dir", type=str, default="data/cloudsen12_1000patches",
                        help="Output directory (default: data/cloudsen12_1000patches)")
    
    args = parser.parse_args()
    
    downloaded, failed = download_cloudsen12(
        num_patches=args.num_patches,
        output_dir=args.output_dir
    )
    
    if failed == 0:
        print("\n✅ All patches downloaded successfully!")
    else:
        print(f"\n⚠️  Completed with {failed} failures")
