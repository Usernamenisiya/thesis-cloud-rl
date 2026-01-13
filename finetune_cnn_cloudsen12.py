"""
Evaluate optimal threshold found on train set, applied to test set
Simple but effective classical baseline
"""

import numpy as np
import rasterio
import glob
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from cnn_inference import load_sentinel2_image, get_cloud_mask

def evaluate_threshold_on_test(threshold):
    """Evaluate a specific threshold on test set"""
    # Load data
    data_dir = 'data/cloudsen12_processed'
    image_files = sorted(glob.glob(f'{data_dir}/*_image.tif'))
    mask_files = sorted(glob.glob(f'{data_dir}/*_mask.tif'))
    
    # 80/20 split
    split_idx = int(0.8 * len(image_files))
    test_image_files = image_files[split_idx:]
    test_mask_files = mask_files[split_idx:]
    
    all_gt = []
    all_cnn = []
    
    for img_path, mask_path in zip(test_image_files, test_mask_files):
        # Load image and get CNN prediction
        image = load_sentinel2_image(img_path)
        cnn_prob = get_cloud_mask(image)
        
        # Load ground truth
        with rasterio.open(mask_path) as src:
            ground_truth = src.read(1)
        
        # Apply threshold
        gt_binary = (ground_truth > 0).astype(np.uint8)
        cnn_binary = (cnn_prob > threshold).astype(np.uint8)
        
        all_gt.append(gt_binary.flatten())
        all_cnn.append(cnn_binary.flatten())
    
    # Combine all patches
    all_gt = np.concatenate(all_gt)
    all_cnn = np.concatenate(all_cnn)
    
    # Calculate metrics
    accuracy = accuracy_score(all_gt, all_cnn)
    precision = precision_score(all_gt, all_cnn, zero_division=0)
    recall = recall_score(all_gt, all_cnn, zero_division=0)
    f1 = f1_score(all_gt, all_cnn, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def main():
    print("🎯 Evaluate Optimal Threshold on Test Set")
    print("="*60)
    
    # Load optimal threshold from train results
    if not Path('results/optimal_threshold_results.json').exists():
        print("❌ No optimal threshold results found. Run optimize_threshold_grid_search.py first.")
        return
    
    with open('results/optimal_threshold_results.json') as f:
        train_results = json.load(f)
    
    optimal_threshold = train_results['best_threshold']
    train_metrics = train_results['train_metrics']
    
    print(f"\n📊 Optimal threshold from training: {optimal_threshold:.2f}")
    print(f"📊 Train F1-Score: {train_metrics['f1_score']:.4f}")
    
    # Evaluate on test set
    print(f"\n🔍 Evaluating on test set...")
    test_metrics = evaluate_threshold_on_test(optimal_threshold)
    
    # Also evaluate baseline (threshold=0.5)
    baseline_metrics = evaluate_threshold_on_test(0.5)
    
    # Calculate improvement
    f1_improvement = ((test_metrics['f1_score'] - baseline_metrics['f1_score']) / baseline_metrics['f1_score'] * 100) if baseline_metrics['f1_score'] > 0 else 0
    
    print("\n" + "="*60)
    print("📈 TEST SET RESULTS")
    print("="*60)
    
    print(f"\n🧠 Baseline CNN (threshold=0.5):")
    print(f"  Accuracy:  {baseline_metrics['accuracy']:.4f}")
    print(f"  Precision: {baseline_metrics['precision']:.4f}")
    print(f"  Recall:    {baseline_metrics['recall']:.4f}")
    print(f"  F1-Score:  {baseline_metrics['f1_score']:.4f}")
    
    print(f"\n🎯 Optimal Threshold ({optimal_threshold:.2f}):")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    
    print(f"\n📊 Improvement:")
    print(f"  F1-Score:  {f1_improvement:+.2f}%")
    print(f"  Accuracy:  {(test_metrics['accuracy'] - baseline_metrics['accuracy']) / baseline_metrics['accuracy'] * 100:+.2f}%")
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    results = {
        'optimal_threshold': optimal_threshold,
        'baseline_threshold': 0.5,
        'train_metrics': train_metrics,
        'test_baseline_metrics': baseline_metrics,
        'test_optimal_metrics': test_metrics,
        'improvement_percent': f1_improvement
    }
    
    with open('results/optimal_threshold_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: results/optimal_threshold_test_results.json")
    print("="*60)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
