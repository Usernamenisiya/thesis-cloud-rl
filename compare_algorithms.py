"""
Compare CNN Baseline, PPO, and DQN on Thin Cloud Detection
Generates comparison table and visualizations for thesis.

Usage:
    python compare_algorithms.py
"""

import os
import sys
import glob
import numpy as np
from datetime import datetime

# Check if running in Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    PPO_MODEL_DIR = '/content/drive/MyDrive/Colab_Data/thin_cloud_v2'
    DQN_MODEL_DIR = '/content/drive/MyDrive/Colab_Data/dqn_thin_cloud'
    DATA_DIR = '/content/drive/MyDrive/Colab_Data/cloudsen12_processed_1000'
    OUTPUT_DIR = '/content/drive/MyDrive/Colab_Data/algorithm_comparison'
else:
    PPO_MODEL_DIR = 'checkpoints/thin_cloud'
    DQN_MODEL_DIR = 'checkpoints/dqn_thin_cloud'
    DATA_DIR = 'data/cloudsen12_processed_1000'
    OUTPUT_DIR = 'results/algorithm_comparison'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_test_data():
    """Load test set (last 20% of data)."""
    import rasterio
    from cnn_inference import load_sentinel2_image, get_cloud_mask
    
    image_files = sorted(glob.glob(f'{DATA_DIR}/*_image.tif'))
    mask_files = sorted(glob.glob(f'{DATA_DIR}/*_mask.tif'))
    
    split_idx = int(0.8 * len(image_files))
    test_images = image_files[split_idx:]
    test_masks = mask_files[split_idx:]
    
    print(f"📊 Loaded {len(test_images)} test patches")
    return test_images, test_masks

def evaluate_baseline(test_images, test_masks, threshold=0.5):
    """Evaluate CNN baseline (s2cloudless)."""
    import rasterio
    from cnn_inference import load_sentinel2_image, get_cloud_mask
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from rl_thin_cloud_environment import ThinCloudDetectionEnv
    from tqdm import tqdm
    
    print("\n🔍 Evaluating CNN Baseline...")
    
    all_gt = []
    all_pred = []
    thin_correct = 0
    thin_total = 0
    
    for img_path, mask_path in tqdm(zip(test_images, test_masks), total=len(test_images)):
        image = load_sentinel2_image(img_path)
        cnn_prob = get_cloud_mask(image)
        
        with rasterio.open(mask_path) as src:
            gt = src.read(1)
        gt_binary = (gt > 0).astype(np.uint8)
        
        pred = (cnn_prob > threshold).astype(np.uint8)
        
        # Thin cloud metrics
        env = ThinCloudDetectionEnv(image, cnn_prob, gt_binary, patch_size=64)
        thin_mask = env.thin_cloud_indicator.astype(bool) & (gt_binary == 1)
        
        if thin_mask.sum() > 0:
            thin_correct += np.sum((pred == 1) & thin_mask)
            thin_total += thin_mask.sum()
        
        all_gt.append(gt_binary.flatten())
        all_pred.append(pred.flatten())
    
    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)
    
    results = {
        'method': 'CNN Baseline',
        'accuracy': accuracy_score(all_gt, all_pred) * 100,
        'precision': precision_score(all_gt, all_pred, zero_division=0) * 100,
        'recall': recall_score(all_gt, all_pred, zero_division=0) * 100,
        'f1': f1_score(all_gt, all_pred, zero_division=0) * 100,
        'thin_recall': (thin_correct / thin_total * 100) if thin_total > 0 else 0
    }
    
    return results

def evaluate_ppo(test_images, test_masks):
    """Evaluate PPO model."""
    import rasterio
    from stable_baselines3 import PPO
    from cnn_inference import load_sentinel2_image, get_cloud_mask
    from rl_thin_cloud_environment import ThinCloudDetectionEnv
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from tqdm import tqdm
    
    # Find latest PPO model
    checkpoints = sorted(glob.glob(f'{PPO_MODEL_DIR}/thin_cloud_*_steps.zip'))
    if not checkpoints:
        print("❌ No PPO model found!")
        return None
    
    model_path = checkpoints[-1]
    print(f"\n🔍 Evaluating PPO: {model_path}")
    
    model = PPO.load(model_path)
    
    all_gt = []
    all_pred = []
    thin_correct = 0
    thin_total = 0
    
    for img_path, mask_path in tqdm(zip(test_images, test_masks), total=len(test_images)):
        image = load_sentinel2_image(img_path)
        cnn_prob = get_cloud_mask(image)
        
        with rasterio.open(mask_path) as src:
            gt = src.read(1)
        gt_binary = (gt > 0).astype(np.uint8)
        
        # Create environment and get prediction
        env = ThinCloudDetectionEnv(image, cnn_prob, gt_binary, patch_size=64)
        rl_pred = np.zeros_like(gt_binary, dtype=np.uint8)
        
        obs, _ = env.reset()
        for _ in range(env.num_patches):
            action, _ = model.predict(obs, deterministic=True)
            
            i, j = env.current_pos
            ps = env.patch_size
            
            threshold_delta = np.clip(action[0], -0.2, 0.2)
            thin_boost = np.clip(action[1], 0.0, 0.3)
            
            cnn_patch = cnn_prob[i:i+ps, j:j+ps].copy()
            thin_indicator = env.thin_cloud_indicator[i:i+ps, j:j+ps]
            
            boosted_prob = np.clip(cnn_patch + thin_indicator * thin_boost, 0, 1)
            rl_pred[i:i+ps, j:j+ps] = (boosted_prob > (0.5 + threshold_delta)).astype(np.uint8)
            
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        
        # Thin cloud metrics
        thin_mask = env.thin_cloud_indicator.astype(bool) & (gt_binary == 1)
        
        if thin_mask.sum() > 0:
            thin_correct += np.sum((rl_pred == 1) & thin_mask)
            thin_total += thin_mask.sum()
        
        all_gt.append(gt_binary.flatten())
        all_pred.append(rl_pred.flatten())
    
    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)
    
    results = {
        'method': 'PPO',
        'accuracy': accuracy_score(all_gt, all_pred) * 100,
        'precision': precision_score(all_gt, all_pred, zero_division=0) * 100,
        'recall': recall_score(all_gt, all_pred, zero_division=0) * 100,
        'f1': f1_score(all_gt, all_pred, zero_division=0) * 100,
        'thin_recall': (thin_correct / thin_total * 100) if thin_total > 0 else 0
    }
    
    return results

def evaluate_dqn(test_images, test_masks):
    """Evaluate DQN model."""
    import rasterio
    from stable_baselines3 import DQN
    from cnn_inference import load_sentinel2_image, get_cloud_mask
    from rl_thin_cloud_environment_discrete import ThinCloudDetectionEnvDiscrete
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from tqdm import tqdm
    
    # Find latest DQN model
    checkpoints = sorted(glob.glob(f'{DQN_MODEL_DIR}/dqn_thin_cloud_*_steps.zip'))
    if not checkpoints:
        print("❌ No DQN model found!")
        return None
    
    model_path = checkpoints[-1]
    print(f"\n🔍 Evaluating DQN: {model_path}")
    
    model = DQN.load(model_path)
    
    all_gt = []
    all_pred = []
    thin_correct = 0
    thin_total = 0
    
    for img_path, mask_path in tqdm(zip(test_images, test_masks), total=len(test_images)):
        image = load_sentinel2_image(img_path)
        cnn_prob = get_cloud_mask(image)
        
        with rasterio.open(mask_path) as src:
            gt = src.read(1)
        gt_binary = (gt > 0).astype(np.uint8)
        
        # Create environment and get prediction
        env = ThinCloudDetectionEnvDiscrete(image, cnn_prob, gt_binary, patch_size=64)
        rl_pred = np.zeros_like(gt_binary, dtype=np.uint8)
        
        obs, _ = env.reset()
        for _ in range(env.num_patches):
            action, _ = model.predict(obs, deterministic=True)
            
            # Decode discrete action
            threshold_delta, thin_boost = env._decode_action(action)
            
            i, j = env.current_pos
            ps = env.patch_size
            
            cnn_patch = cnn_prob[i:i+ps, j:j+ps].copy()
            thin_indicator = env.thin_cloud_indicator[i:i+ps, j:j+ps]
            
            boosted_prob = np.clip(cnn_patch + thin_indicator * thin_boost, 0, 1)
            rl_pred[i:i+ps, j:j+ps] = (boosted_prob > (0.5 + threshold_delta)).astype(np.uint8)
            
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        
        # Thin cloud metrics
        thin_mask = env.thin_cloud_indicator.astype(bool) & (gt_binary == 1)
        
        if thin_mask.sum() > 0:
            thin_correct += np.sum((rl_pred == 1) & thin_mask)
            thin_total += thin_mask.sum()
        
        all_gt.append(gt_binary.flatten())
        all_pred.append(rl_pred.flatten())
    
    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)
    
    results = {
        'method': 'DQN',
        'accuracy': accuracy_score(all_gt, all_pred) * 100,
        'precision': precision_score(all_gt, all_pred, zero_division=0) * 100,
        'recall': recall_score(all_gt, all_pred, zero_division=0) * 100,
        'f1': f1_score(all_gt, all_pred, zero_division=0) * 100,
        'thin_recall': (thin_correct / thin_total * 100) if thin_total > 0 else 0
    }
    
    return results

def create_comparison_table(results_list):
    """Create and print comparison table."""
    print("\n" + "=" * 80)
    print("📊 ALGORITHM COMPARISON: CNN vs PPO vs DQN")
    print("=" * 80)
    
    header = f"{'Method':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Thin Recall':>12}"
    print(header)
    print("-" * 80)
    
    for r in results_list:
        if r is not None:
            row = f"{r['method']:<15} {r['accuracy']:>9.2f}% {r['precision']:>9.2f}% {r['recall']:>9.2f}% {r['f1']:>9.2f}% {r['thin_recall']:>11.2f}%"
            print(row)
    
    print("=" * 80)
    
    return results_list

def create_comparison_plot(results_list):
    """Create bar chart comparing all methods."""
    import matplotlib.pyplot as plt
    
    # Filter out None results
    results_list = [r for r in results_list if r is not None]
    
    if len(results_list) < 2:
        print("⚠️ Need at least 2 methods to compare")
        return
    
    methods = [r['method'] for r in results_list]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Thin Cloud\nRecall']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: All metrics
    x = np.arange(len(metrics))
    width = 0.25
    colors = ['#3498db', '#2ecc71', '#e74c3c']  # Blue, Green, Red
    
    for i, r in enumerate(results_list):
        values = [r['accuracy'], r['precision'], r['recall'], r['f1'], r['thin_recall']]
        offset = (i - len(results_list)/2 + 0.5) * width
        bars = axes[0].bar(x + offset, values, width, label=r['method'], color=colors[i], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[0].annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', fontsize=8)
    
    axes[0].set_ylabel('Percentage (%)', fontsize=12)
    axes[0].set_title('Overall Metrics Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].set_ylim(0, 105)
    
    # Right: Thin cloud recall focus
    thin_values = [r['thin_recall'] for r in results_list]
    bars = axes[1].bar(methods, thin_values, color=colors[:len(methods)], alpha=0.8, 
                       edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1].annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', fontsize=14, fontweight='bold')
    
    axes[1].set_ylabel('Thin Cloud Recall (%)', fontsize=12)
    axes[1].set_title('🎯 Key Metric: Thin Cloud Detection', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, max(thin_values) * 1.2)
    
    plt.suptitle('Algorithm Comparison: CNN Baseline vs PPO vs DQN', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    output_path = f'{OUTPUT_DIR}/algorithm_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved to: {output_path}")
    plt.show()

def main():
    print("=" * 70)
    print("🔬 ALGORITHM COMPARISON: CNN vs PPO vs DQN")
    print("=" * 70)
    print(f"   Thin Cloud Detection Task")
    print(f"   Dataset: CloudSEN12 (1000 patches)")
    print("=" * 70)
    
    # Load test data
    test_images, test_masks = load_test_data()
    
    # Evaluate all methods
    results = []
    
    # 1. CNN Baseline
    baseline_results = evaluate_baseline(test_images, test_masks)
    results.append(baseline_results)
    
    # 2. PPO
    ppo_results = evaluate_ppo(test_images, test_masks)
    results.append(ppo_results)
    
    # 3. DQN
    dqn_results = evaluate_dqn(test_images, test_masks)
    results.append(dqn_results)
    
    # Create comparison
    create_comparison_table(results)
    create_comparison_plot(results)
    
    # Save results to JSON
    import json
    with open(f'{OUTPUT_DIR}/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {OUTPUT_DIR}/comparison_results.json")

if __name__ == "__main__":
    main()
