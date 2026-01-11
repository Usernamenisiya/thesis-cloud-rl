#!/usr/bin/env python3
"""
Data setup utilities for cloud storage integration.
This script provides functions to download data from various cloud storage providers.
"""

import os
import requests
from pathlib import Path
import subprocess
import sys

def download_from_dropbox(dropbox_link, output_path):
    """
    Download file from Dropbox shared link.

    Args:
        dropbox_link: Dropbox share link (should end with ?dl=0 or ?dl=1)
        output_path: Local path to save the file
    """
    # Convert share link to direct download link
    if '?dl=0' in dropbox_link:
        direct_link = dropbox_link.replace('?dl=0', '?dl=1')
    elif '?dl=1' not in dropbox_link:
        direct_link = dropbox_link + '?dl=1'
    else:
        direct_link = dropbox_link

    print(f"Downloading from Dropbox: {direct_link}")
    print(f"Saving to: {output_path}")

    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download file
    response = requests.get(direct_link, stream=True)
    response.raise_for_status()

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"✅ Downloaded: {output_path}")

def download_from_onedrive(onedrive_link, output_path):
    """
    Download file from OneDrive shared link.

    Args:
        onedrive_link: OneDrive share link
        output_path: Local path to save the file
    """
    print(f"Downloading from OneDrive: {onedrive_link}")
    print(f"Saving to: {output_path}")

    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use wget to download (works better with OneDrive redirects)
    try:
        result = subprocess.run([
            'wget', '-O', str(output_path), onedrive_link
        ], capture_output=True, text=True, check=True)
        print(f"✅ Downloaded: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print(f"Error output: {e.stderr}")

def download_from_google_drive(gdrive_id, output_path):
    """
    Download file from Google Drive.

    Args:
        gdrive_id: Google Drive file ID (from share link)
        output_path: Local path to save the file
    """
    print(f"Downloading from Google Drive (ID: {gdrive_id})")
    print(f"Saving to: {output_path}")

    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Google Drive download URL
    url = f"https://drive.google.com/uc?id={gdrive_id}&export=download"

    try:
        result = subprocess.run([
            'wget', '--no-check-certificate', '-O', str(output_path), url
        ], capture_output=True, text=True, check=True)
        print(f"✅ Downloaded: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print(f"Error output: {e.stderr}")

def setup_data_from_cloud():
    """
    Interactive setup for downloading data from cloud storage.
    """
    print("🌐 Cloud Data Setup")
    print("==================")

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Check what files we need
    sentinel_path = data_dir / "sentinel2_image.tif"
    ground_truth_path = data_dir / "ground_truth.tif"

    needed_files = []
    if not sentinel_path.exists():
        needed_files.append(("Sentinel-2 image", sentinel_path))
    if not ground_truth_path.exists():
        needed_files.append(("Ground truth", ground_truth_path))

    if not needed_files:
        print("✅ All data files already exist!")
        return

    print(f"📁 Need to download {len(needed_files)} file(s):")
    for name, path in needed_files:
        print(f"  - {name}: {path}")

    print("\nChoose your cloud storage provider:")
    print("1. Dropbox")
    print("2. OneDrive")
    print("3. Google Drive")
    print("4. Direct URL (any other source)")

    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()

            if choice == "1":
                print("\n📦 Dropbox Setup")
                print("1. Right-click your file in Dropbox web")
                print("2. Select 'Share' > 'Create link'")
                print("3. Copy the link (should end with ?dl=0)")
                print("4. Paste it below:")

                for name, path in needed_files:
                    link = input(f"\nEnter Dropbox link for {name}: ").strip()
                    if link:
                        download_from_dropbox(link, path)

            elif choice == "2":
                print("\n📦 OneDrive Setup")
                print("1. Right-click your file in OneDrive web")
                print("2. Select 'Share' > 'Create link'")
                print("3. Copy the link and paste it below:")

                for name, path in needed_files:
                    link = input(f"\nEnter OneDrive link for {name}: ").strip()
                    if link:
                        download_from_onedrive(link, path)

            elif choice == "3":
                print("\n📦 Google Drive Setup")
                print("1. Right-click your file in Google Drive")
                print("2. Select 'Get shareable link'")
                print("3. Copy the file ID from the link (the long string between /d/ and /view)")
                print("   Example: https://drive.google.com/file/d/FILE_ID_HERE/view")

                for name, path in needed_files:
                    file_id = input(f"\nEnter Google Drive file ID for {name}: ").strip()
                    if file_id:
                        download_from_google_drive(file_id, path)

            elif choice == "4":
                print("\n📦 Direct URL Setup")
                print("Enter any direct download URL:")

                for name, path in needed_files:
                    url = input(f"\nEnter direct URL for {name}: ").strip()
                    if url:
                        print(f"Downloading from: {url}")
                        print(f"Saving to: {path}")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        subprocess.run(['wget', '-O', str(path), url], check=True)
                        print(f"✅ Downloaded: {path}")

            else:
                print("❌ Invalid choice. Please enter 1-4.")
                continue

            break

        except KeyboardInterrupt:
            print("\n❌ Setup cancelled.")
            return
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    print("\n✅ Data setup complete!")
    print("Run the data verification cell in your notebook to confirm everything works.")

if __name__ == "__main__":
    setup_data_from_cloud()