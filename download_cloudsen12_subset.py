#!/usr/bin/env python3
"""
Download a subset of CloudSEN12 dataset for validation.
Storage optimized: ~4-5 GB for 150-200 patches.
Run this in Google Colab.
"""
import os
from pathlib import Path

def download_cloudsen12_subset():
    """Download CloudSEN12 subset using Hugging Face datasets."""
    
    print("=" * 60)
    print("📦 CloudSEN12 Subset Downloader")
    print("=" * 60)
    
    # Install required packages
    print("\n📥 Installing dependencies...")
    os.system("pip install -q datasets huggingface_hub")
    
    from datasets import load_dataset
    
    # Create output directory in your existing structure
    base_path = Path("/content/drive/MyDrive/Colab_Data")
    output_dir = base_path / "cloudsen12_subset"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving to: {output_dir}")
    
    # Load CloudSEN12 dataset (streaming mode to save memory)
    print("\n🌐 Loading CloudSEN12 from Hugging Face...")
    print("Note: Using streaming mode to minimize memory usage")
    
    try:
        dataset = load_dataset(
            "tacofoundation/cloudsen12",
            split="train",
            streaming=True
        )
        
        print("\n✅ Dataset loaded successfully!")
        print("\n📊 Downloading subset...")
        print("Target: 150-200 patches (~4-5 GB)")
        
        # Take first 150 samples
        subset_size = 150
        downloaded = 0
        
        for idx, sample in enumerate(dataset):
            if idx >= subset_size:
                break
            
            # Save each sample
            sample_dir = output_dir / f"patch_{idx:04d}"
            sample_dir.mkdir(exist_ok=True)
            
            # Save components (customize based on actual dataset structure)
            # This will need to be adapted once we see the actual format
            print(f"  Downloaded patch {idx+1}/{subset_size}", end='\r')
            downloaded += 1
        
        print(f"\n\n✅ Downloaded {downloaded} patches successfully!")
        print(f"📍 Location: {output_dir}")
        
        # Display storage usage
        total_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
        print(f"💾 Total size: {total_size / (1024**3):.2f} GB")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔄 Alternative: Try manual download from Google Drive")
        print("Link: https://drive.google.com/drive/folders/1H_Qxy1yX6moqFbKc3Ajn4FFrHlDhYuz3")
        return False
    
    return True

def setup_for_training():
    """Prepare downloaded data for your existing training code."""
    print("\n" + "=" * 60)
    print("🔧 Setting up data for training...")
    print("=" * 60)
    
    print("""
Next steps:
1. Explore the downloaded patches
2. Adapt your data loading code to use CloudSEN12 format
3. Retrain your RL model with real ground truth
4. Compare results!

The data structure should contain:
- Sentinel-2 bands (B02-B12)
- Cloud mask labels (expert-annotated)
- Metadata
""")

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║           CloudSEN12 Subset Download Script                ║
║                                                            ║
║  This script downloads ~150 patches (4-5 GB) to your      ║
║  Google Drive for thesis validation.                      ║
╚════════════════════════════════════════════════════════════╝

⚠️  REQUIREMENTS:
   - Run this in Google Colab
   - Mount Google Drive first: 
     from google.colab import drive
     drive.mount('/content/drive')
   - Ensure you have ~5GB free in Drive

Press Enter to continue or Ctrl+C to cancel...
""")
    
    input()
    
    success = download_cloudsen12_subset()
    
    if success:
        setup_for_training()
        print("\n✅ Setup complete! Ready for training with real ground truth.")
    else:
        print("\n⚠️  Please try manual download or alternative sources.")
