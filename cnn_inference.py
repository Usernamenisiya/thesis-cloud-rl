import numpy as np
import cv2
from s2cloudless import S2PixelCloudDetector
import matplotlib.pyplot as plt
import rasterio

# Initialize the cloud detector
cloud_detector = S2PixelCloudDetector(threshold=0.4, all_bands=False, average_over=4)

def load_sentinel2_image(image_path):
    """
    Load a Sentinel-2 image using rasterio.
    Select the 10 bands used by s2cloudless: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
    Assuming bands are in order B02 to B12 (indices 0 to 9).
    """
    with rasterio.open(image_path) as src:
        # Read all bands
        all_bands = src.read()  # shape (10, H, W)
        # All bands are required in order
        image = all_bands  # (10, H, W)
        # Transpose to (H, W, 10) for consistency
        image = np.transpose(image, (1, 2, 0)).astype(np.float32)
    return image

def get_cloud_mask(image):
    """
    Get cloud probability mask from CNN.
    Image is (H, W, 10)
    """
    # s2cloudless expects (batch, H, W, bands)
    image_batched = image[np.newaxis, ...]  # to (1, H, W, 10)
    cloud_prob = cloud_detector.get_cloud_probability_maps(image_batched)
    # cloud_prob is (1, H, W), so squeeze to (H, W)
    return cloud_prob[0]

def binarize_mask(prob_mask, threshold=0.5):
    """
    Binarize probability mask to 0/1.
    """
    return (prob_mask > threshold).astype(np.uint8)

# Example usage
if __name__ == "__main__":
    image_path = "data/sentinel2_image.tif"  # replace
    image = load_sentinel2_image(image_path)
    prob_mask = get_cloud_mask(image)
    binary_mask = binarize_mask(prob_mask)

    plt.imshow(binary_mask, cmap='gray')
    plt.title('Cloud Mask')
    plt.show()