"""
CloudSEN12 Pretrained CNN Inference

Uses maskay library's pretrained models specifically trained on CloudSEN12.
These models should perform significantly better than s2cloudless on thin clouds.

Available models:
- UnetMobV2: U-Net with MobileNetV2 backbone (recommended)
- KappaModelUNetL1C: Kappa U-Net for L1C data
- CDFCNNrgbi: CDF-CNN for RGBI bands

Author: Thesis Implementation
Date: January 2026
"""

import numpy as np
import rasterio
import torch

# Import maskay models
try:
    from maskay import Predictor, TensorSat
    from maskay.library.unetmobv2 import UnetMobV2
    MASKAY_AVAILABLE = True
except ImportError:
    MASKAY_AVAILABLE = False
    print("⚠️  maskay not installed. Install with: pip install maskay")


def load_sentinel2_image(image_path):
    """
    Load a Sentinel-2 image using rasterio.
    Returns (H, W, 10) array with all bands.
    """
    with rasterio.open(image_path) as src:
        all_bands = src.read()  # shape (10, H, W)
        image = np.transpose(all_bands, (1, 2, 0)).astype(np.float32)
    return image


def get_cloud_mask_cloudsen12(image, model_type='unet', device='cpu'):
    """
    Get cloud probability mask using CloudSEN12 pretrained models.
    
    Args:
        image: (H, W, 10) Sentinel-2 image
        model_type: 'unet' (UnetMobV2) or 'cdfcnn' (CDFCNNrgbi)
        device: 'cpu' or 'cuda'
    
    Returns:
        cloud_prob: (H, W) probability map [0-1]
    """
    if not MASKAY_AVAILABLE:
        raise ImportError("maskay library is required. Install with: pip install maskay")
    
    # Load pretrained model
    if model_type == 'unet':
        model = UnetMobV2()
        print("✅ Loaded UnetMobV2 (CloudSEN12 pretrained)")
    elif model_type == 'cdfcnn':
        from maskay.library.cdfcnnrgbi import CDFCNNrgbi
        model = CDFCNNrgbi()
        print("✅ Loaded CDFCNNrgbi (CloudSEN12 pretrained)")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Create predictor
    predictor = Predictor(
        cropsize=512,
        overlap=32,
        device=device,
        batchsize=1 if device == 'cpu' else 4,
        quiet=True,
        order='BHWC'
    )
    
    # Convert to TensorSat format (expecting channels-first)
    # maskay expects (C, H, W) format
    image_chw = np.transpose(image, (2, 0, 1))
    
    # Create a minimal TensorSat-like object
    # Note: This is a simplified approach - adjust based on maskay requirements
    tensor_data = torch.from_numpy(image_chw).float()
    
    # Predict
    result = predictor.predict(model, tensor_data)
    
    # Extract cloud probability
    # maskay typically returns classification logits or probabilities
    # Adjust based on actual output format
    if isinstance(result, torch.Tensor):
        cloud_prob = result.cpu().numpy()
    else:
        cloud_prob = result
    
    # Ensure output is (H, W) with values in [0, 1]
    if cloud_prob.ndim == 3:
        # If multi-class, take cloud class (usually index 1)
        if cloud_prob.shape[0] > 1:
            cloud_prob = cloud_prob[1]  # Cloud class
        else:
            cloud_prob = cloud_prob[0]
    
    # Normalize to [0, 1] if needed
    if cloud_prob.max() > 1.0:
        cloud_prob = cloud_prob / 255.0
    
    return cloud_prob


def get_cloud_mask_simple(image, device='cpu'):
    """
    Simplified inference using CloudSEN12 UnetMobV2 model.
    This is a more robust implementation.
    
    Args:
        image: (H, W, 10) Sentinel-2 image
        device: 'cpu' or 'cuda'
    
    Returns:
        cloud_prob: (H, W) probability map [0-1]
    """
    if not MASKAY_AVAILABLE:
        raise ImportError("maskay library is required. Install with: pip install maskay")
    
    # Use UnetMobV2 (most reliable CloudSEN12 model)
    model = UnetMobV2()
    model.eval()
    
    if device == 'cuda':
        model = model.cuda()
    
    # Convert to tensor (C, H, W)
    image_chw = np.transpose(image, (2, 0, 1))
    image_tensor = torch.from_numpy(image_chw).float().unsqueeze(0)  # (1, C, H, W)
    
    if device == 'cuda':
        image_tensor = image_tensor.cuda()
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
    
    # Extract cloud probability
    # Assuming binary classification (cloud vs non-cloud)
    if output.shape[1] > 1:
        # Multi-class: take cloud class (index 1)
        cloud_prob = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
    else:
        # Binary: apply sigmoid
        cloud_prob = torch.sigmoid(output[0, 0]).cpu().numpy()
    
    return cloud_prob


def binarize_mask(prob_mask, threshold=0.5):
    """
    Binarize probability mask to 0/1.
    """
    return (prob_mask > threshold).astype(np.uint8)


def compare_models(image_path, ground_truth_path=None):
    """
    Compare s2cloudless vs CloudSEN12 models.
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON: s2cloudless vs CloudSEN12")
    print("="*80)
    
    # Load image
    image = load_sentinel2_image(image_path)
    print(f"✅ Loaded image: {image.shape}")
    
    # s2cloudless
    try:
        from cnn_inference import get_cloud_mask as get_cloud_mask_s2cloudless
        s2_prob = get_cloud_mask_s2cloudless(image)
        print(f"✅ s2cloudless prediction: {s2_prob.shape}")
    except Exception as e:
        print(f"❌ s2cloudless failed: {e}")
        s2_prob = None
    
    # CloudSEN12
    try:
        cs12_prob = get_cloud_mask_simple(image, device='cpu')
        print(f"✅ CloudSEN12 prediction: {cs12_prob.shape}")
    except Exception as e:
        print(f"❌ CloudSEN12 failed: {e}")
        cs12_prob = None
    
    # Compare if ground truth available
    if ground_truth_path and s2_prob is not None and cs12_prob is not None:
        with rasterio.open(ground_truth_path) as src:
            gt = (src.read(1) > 0).astype(np.uint8)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        s2_binary = binarize_mask(s2_prob, 0.5)
        cs12_binary = binarize_mask(cs12_prob, 0.5)
        
        print("\n📊 METRICS COMPARISON:")
        print("-" * 80)
        print(f"{'Metric':<15} {'s2cloudless':<15} {'CloudSEN12':<15} {'Improvement':<15}")
        print("-" * 80)
        
        for name, metric_fn in [
            ('Accuracy', accuracy_score),
            ('Precision', lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0)),
            ('Recall', lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0)),
            ('F1-Score', lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0))
        ]:
            s2_val = metric_fn(gt.flatten(), s2_binary.flatten())
            cs12_val = metric_fn(gt.flatten(), cs12_binary.flatten())
            improvement = ((cs12_val - s2_val) / (s2_val + 1e-8)) * 100
            
            print(f"{name:<15} {s2_val:<15.4f} {cs12_val:<15.4f} {improvement:+.2f}%")
        
        print("-" * 80)
    
    print("\n" + "="*80)
    
    return s2_prob, cs12_prob


# Example usage
if __name__ == "__main__":
    import glob
    
    # Test on first patch
    data_dir = "data/cloudsen12_processed"
    images = glob.glob(f"{data_dir}/*_image.tif")
    masks = glob.glob(f"{data_dir}/*_mask.tif")
    
    if images and masks:
        print(f"Testing on: {images[0]}")
        compare_models(images[0], masks[0])
    else:
        print("No test data found. Please specify image path manually.")
