"""
Grid search to find optimal CNN threshold on train set (80 patches)
Fast classical approach - no training required
"""

import glob
import numpy as np
import rasterio
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from cnn_inference import load_sentinel2_image, get_cloud_mask
import json
from pathlib import Path

def evaluate_threshold(all_cnn_probs, all_gt, threshold):
    """Evaluate a single threshold value"""
    all_cnn_binary = [(prob > threshold).astype(np.uint8) for prob in all_cnn_probs]
    all_cnn_flat = np.concatenate([pred.flatten() for pred in all_cnn_binary])
    all_gt_flat = np.concatenate([gt.flatten() for gt in all_gt])
    
    accuracy = accuracy_score(all_gt_flat, all_cnn_flat)
    precision = precision_score(all_gt_flat, all_cnn_flat, zero_division=0)
    recall = recall_score(all_gt_flat, all_cnn_flat, zero_division=0)
    f1 = f1_score(all_gt_flat, all_cnn_flat, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def main():
    print("🔍 Optimal Threshold Grid Search")
    print("="*60)
    
    # Load train set (first 80 patches)
    data_dir = 'data/cloudsen12_processed'
    image_files = sorted(glob.glob(f'{data_dir}/*_image.tif'))
    mask_files = sorted(glob.glob(f'{data_dir}/*_mask.tif'))
    
    split_idx = int(0.8 * len(image_files))
    train_image_files = image_files[:split_idx]
    train_mask_files = mask_files[:split_idx]
    
    print(f"📊 Loading {len(train_image_files)} training patches...")
    
    # Load all CNN probabilities and ground truth
    all_cnn_probs = []
    all_gt = []
    
    for idx, (img_path, mask_path) in enumerate(zip(train_image_files, train_mask_files)):
        if (idx + 1) % 10 == 0:
            print(f"  Loaded {idx+1}/{len(train_image_files)} patches", end='\r')
        
        image = load_sentinel2_image(img_path)
        cnn_prob = get_cloud_mask(image)
        
        with rasterio.open(mask_path) as src:
            ground_truth = src.read(1)
        
        gt_binary = (ground_truth > 0).astype(np.uint8)
        
        all_cnn_probs.append(cnn_prob)
        all_gt.append(gt_binary)
    
    print(f"\n✅ Loaded {len(all_cnn_probs)} training patches")
    
    # Grid search over thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    print(f"\n🔍 Testing {len(thresholds)} thresholds...")
    
    results = []
    for threshold in thresholds:
        metrics = evaluate_threshold(all_cnn_probs, all_gt, threshold)
        metrics['threshold'] = float(threshold)
        results.append(metrics)
        print(f"  Threshold {threshold:.2f}: F1={metrics['f1_score']:.4f}, Acc={metrics['accuracy']:.4f}")
    
    # Find best threshold by F1-score
    best_result = max(results, key=lambda x: x['f1_score'])
    best_threshold = best_result['threshold']
    
    print("\n" + "="*60)
    print("🎯 OPTIMAL THRESHOLD RESULTS (Train Set)")
    print("="*60)
    print(f"\n✨ Best Threshold: {best_threshold:.2f}")
    print(f"\n📊 Performance:")
    print(f"  Accuracy:  {best_result['accuracy']:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall:    {best_result['recall']:.4f}")
    print(f"  F1-Score:  {best_result['f1_score']:.4f}")
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    output = {
        'best_threshold': best_threshold,
        'train_metrics': best_result,
        'all_thresholds': results
    }
    
    with open('results/optimal_threshold_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: results/optimal_threshold_results.json")
    print("="*60)
    
    return best_threshold

if __name__ == "__main__":
    main()
