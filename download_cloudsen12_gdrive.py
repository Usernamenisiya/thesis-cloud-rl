"""
Download CloudSEN12 dataset directly to Google Drive using tacoreader.

Run this in Google Colab to download 1000 patches (~25GB) to Google Drive.
Uses official tacoreader method with cloud-optimized streaming.

Output: /content/drive/MyDrive/Colab_Data/cloudsen12_1000patches/
"""

import pathlib

def download_cloudsen12_to_drive(num_patches=1000, 
                                  drive_base="/content/drive/MyDrive/Colab_Data",
                                  output_folder="cloudsen12_1000patches"):
    """
    Download CloudSEN12 patches directly to Google Drive.
    
    Args:
        num_patches: Number of patches to download (default 1000)
        drive_base: Base Google Drive path
        output_folder: Folder name for downloaded data
    """
    print("=" * 80)
    print("📦 CloudSEN12 Dataset Downloader → Google Drive")
    print("=" * 80)
    print(f"\nTarget: {num_patches} patches (~{num_patches * 25 / 1000:.1f} GB)")
    print(f"Output: {drive_base}/{output_folder}")
    print("\n" + "=" * 80)
    
    # Check if Drive is mounted
    import os
    if not os.path.exists(drive_base):
        print("\n❌ Google Drive not mounted!")
        print("Please run first:")
        print("  from google.colab import drive")
        print("  drive.mount('/content/drive')")
        return
    
    # Install tacoreader
    print("\n📥 Installing dependencies...")
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "tacoreader", "rasterio"],
        capture_output=True
    )
    
    import tacoreader
    import rasterio as rio
    
    print(f"✅ tacoreader version: {tacoreader.__version__}")
    
    # Load dataset
    print("\n🌐 Loading CloudSEN12 dataset (streaming mode)...")
    print("Note: Metadata only - actual data downloaded on-demand")
    
    try:
        dataset = tacoreader.load([
            "cloudsen12-l1c.0000.part.taco",
            "cloudsen12-l1c.0001.part.taco",
            "cloudsen12-l1c.0002.part.taco",
            "cloudsen12-l1c.0003.part.taco"
        ])
    except Exception as e:
        print(f"\n❌ Failed to load dataset: {e}")
        print("This might be a network issue. Try again.")
        return
    
    total_available = len(dataset)
    print(f"✅ Dataset loaded: {total_available:,} patches available")
    
    if num_patches > total_available:
        print(f"⚠️  Requested {num_patches} patches but only {total_available} available")
        print(f"   Downloading all {total_available} patches instead")
        num_patches = total_available
    
    # Create output folder in Drive
    out_folder = pathlib.Path(drive_base) / output_folder
    out_folder.mkdir(exist_ok=True, parents=True)
    
    print(f"\n📥 Downloading {num_patches} patches to Google Drive...")
    print(f"💾 Saving to: {out_folder}")
    print("\nThis will take 15-45 minutes depending on network speed")
    print("Progress updates every 50 patches\n")
    
    # Download patches
    downloaded = 0
    failed = 0
    
    for index in range(num_patches):
        try:
            # Progress update
            if (index + 1) % 50 == 0:
                print(f"  📊 Progress: {index + 1}/{num_patches} patches ({(index + 1) / num_patches * 100:.1f}%)")
            
            # Read sample
            image = dataset.read(index)
            
            # Create output paths
            patch_id = dataset.iloc[index]["tortilla:id"]
            s2l1c_output = out_folder / patch_id / "s2l1c.tif"
            target_output = out_folder / patch_id / "target.tif"
            target_output.parent.mkdir(exist_ok=True, parents=True)
            
            # Skip if already downloaded
            if s2l1c_output.exists() and target_output.exists():
                downloaded += 1
                continue
            
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
            print(f"  ⚠️  Failed patch {index}: {str(e)[:50]}")
            failed += 1
            # Don't stop on individual failures
            if failed > 50:  # But stop if too many failures
                print(f"\n❌ Too many failures ({failed}). Stopping.")
                break
            continue
    
    print("\n" + "=" * 80)
    print("✅ Download Complete!")
    print("=" * 80)
    print(f"Successfully downloaded: {downloaded} patches")
    if failed > 0:
        print(f"Failed: {failed} patches")
    print(f"\n📂 Location: {out_folder}")
    print(f"\nFolder structure:")
    print(f"  {output_folder}/")
    print(f"    ROI_XXXX_<timestamp>_<tile>/")
    print(f"      s2l1c.tif   (Sentinel-2 L1C image - 10 bands)")
    print(f"      target.tif  (Cloud mask labels)")
    print("\n💡 Use this path in your training scripts:")
    print(f"   data_dir = '/content/drive/MyDrive/Colab_Data/{output_folder}'")
    print("\n" + "=" * 80)
    
    return downloaded, failed


# For running in Colab notebook
if __name__ == "__main__":
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
    
    if IN_COLAB:
        # Mount Drive if not already mounted
        from google.colab import drive
        import os
        
        if not os.path.exists('/content/drive'):
            print("📁 Mounting Google Drive...")
            drive.mount('/content/drive')
        
        # Download 1000 patches
        downloaded, failed = download_cloudsen12_to_drive(num_patches=1000)
        
    else:
        print("⚠️  This script is designed for Google Colab")
        print("Please run in Colab or use download_cloudsen12_local.py")
