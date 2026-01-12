#!/usr/bin/env python3
"""
Optimize CNN threshold for best F1-score on CloudSEN12 data.
This ensures fair baseline comparison with RL model.
"""
import numpy as np
import rasterio
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from cnn_inference import load_sentinel2_image, get_cloud_mask

def evaluate_threshold(cnn_probs, ground_truths, threshold):
    """Evaluate CNN at specific threshold."""
    all_preds = []
    all_gt = []
    
    for cnn_prob, gt in zip(cnn_probs, ground_truths):
        pred = (cnn_prob > threshold).astype(np.uint8)
        all_preds.append(pred.flatten())
        all_gt.append(gt.flatten())
    
    all_preds = np.concatenate(all_preds)
    all_gt = np.concatenate(all_gt)
    
    accuracy = accuracy_score(all_gt, all_preds)
    precision = precision_score(all_gt, all_preds, zero_division=0)
    recall = recall_score(all_gt, all_preds, zero_division=0)
    f1 = f1_score(all_gt, all_preds, zero_division=0)
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def optimize_cnn_threshold():
    """Find optimal CNN threshold on CloudSEN12 data."""
    print("=" * 60)
    print("🔍 Optimizing CNN Threshold on CloudSEN12")
    print("=" * 60)
    
    # Load CloudSEN12 data
    image_files = sorted(glob.glob('data/cloudsen12_processed/*_image.tif'))
    mask_files = sorted(glob.glob('data/cloudsen12_processed/*_mask.tif'))
    
    if len(image_files) == 0:
        raise FileNotFoundError("No CloudSEN12 data found. Run cloudsen12_loader.py first.")
    
    print(f"\n📂 Loading {len(image_files)} CloudSEN12 patches...")
    
    # Get CNN predictions
    cnn_probs = []
    ground_truths = []
    
    for img_path, mask_path in zip(image_files, mask_files):
        image = load_sentinel2_image(img_path)
        cnn_prob = get_cloud_mask(image)
        
        with rasterio.open(mask_path) as src:
            gt = src.read(1)
        
        cnn_probs.append(cnn_prob)
        ground_truths.append(gt)
    
    print("✅ Data loaded")
    
    # Test thresholds from 0.1 to 0.9
    print("\n🔬 Testing thresholds...")
    thresholds = np.arange(0.1, 0.91, 0.05)
    results = []
    
    for threshold in thresholds:
        result = evaluate_threshold(cnn_probs, ground_truths, threshold)
        results.append(result)
        print(f"  Threshold {threshold:.2f}: F1={result['f1_score']:.4f}, "
              f"Acc={result['accuracy']:.4f}, "
              f"Prec={result['precision']:.4f}, "
              f"Rec={result['recall']:.4f}")
    
    # Find best threshold by F1-score
    best_result = max(results, key=lambda x: x['f1_score'])
    
    print("\n" + "=" * 60)
    print("🎯 OPTIMAL CNN BASELINE")
    print("=" * 60)
    print(f"\n✅ Best threshold: {best_result['threshold']:.2f}")
    print(f"\n🧠 Optimized CNN Performance:")
    print(f"  Accuracy:  {best_result['accuracy']:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall:    {best_result['recall']:.4f}")
    print(f"  F1-Score:  {best_result['f1_score']:.4f}")
    
    # Save optimal threshold
    import json
    with open('results/optimal_cnn_threshold.json', 'w') as f:
        json.dump({
            'optimal_threshold': best_result['threshold'],
            'metrics': best_result,
            'all_results': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: results/optimal_cnn_threshold.json")
    print("\n" + "=" * 60)
    
    return best_result

if __name__ == "__main__":
    optimize_cnn_threshold()
