"""
Convert CloudSEN12 format to training script format.

Converts from:
  ROI_XXXXX__<timestamp>_<tile>/s2l1c.tif
  ROI_XXXXX__<timestamp>_<tile>/target.tif

To flat structure:
  ROI_XXXXX__<timestamp>_<tile>_image.tif
  ROI_XXXXX__<timestamp>_<tile>_mask.tif

Run in Colab after downloading patches.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def convert_cloudsen12_format(input_dir, output_dir):
    """
    Convert CloudSEN12 nested format to flat training format.
    
    Args:
        input_dir: CloudSEN12 download directory (with ROI folders)
        output_dir: Output directory for flat structure
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🔄 Converting CloudSEN12 Format")
    print("=" * 80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()
    
    # Find all ROI folders
    roi_folders = [d for d in input_path.iterdir() if d.is_dir()]
    print(f"📊 Found {len(roi_folders)} patches to convert\n")
    
    converted = 0
    failed = 0
    
    for roi_folder in tqdm(roi_folders, desc="Converting"):
        try:
            # Get patch name
            patch_name = roi_folder.name
            
            # Source files
            s2l1c_src = roi_folder / "s2l1c.tif"
            target_src = roi_folder / "target.tif"
            
            # Check files exist
            if not s2l1c_src.exists() or not target_src.exists():
                print(f"  ⚠️  Skipping {patch_name}: missing files")
                failed += 1
                continue
            
            # Destination files (flat structure)
            image_dst = output_path / f"{patch_name}_image.tif"
            mask_dst = output_path / f"{patch_name}_mask.tif"
            
            # Copy files
            shutil.copy2(s2l1c_src, image_dst)
            shutil.copy2(target_src, mask_dst)
            
            converted += 1
            
        except Exception as e:
            print(f"  ❌ Failed {roi_folder.name}: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print("✅ Conversion Complete!")
    print("=" * 80)
    print(f"Successfully converted: {converted} patches")
    if failed > 0:
        print(f"Failed: {failed} patches")
    print(f"\nOutput directory: {output_path}")
    print(f"\nFile format:")
    print(f"  ROI_XXXXX__<timestamp>_<tile>_image.tif")
    print(f"  ROI_XXXXX__<timestamp>_<tile>_mask.tif")
    print("\n" + "=" * 80)
    
    return converted, failed


if __name__ == "__main__":
    # For Colab
    input_dir = "/content/drive/MyDrive/Colab_Data/cloudsen12_1000patches"
    output_dir = "/content/drive/MyDrive/Colab_Data/cloudsen12_processed_1000"
    
    print("\n🚀 Starting conversion...")
    print("This will take 5-10 minutes\n")
    
    converted, failed = convert_cloudsen12_format(input_dir, output_dir)
    
    if failed == 0:
        print(f"\n✅ All {converted} patches converted successfully!")
        print(f"\n💡 Update your training script data_dir to:")
        print(f"   data_dir = '{output_dir}'")
    else:
        print(f"\n⚠️  Completed with {failed} failures")
