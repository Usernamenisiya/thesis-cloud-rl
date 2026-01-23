import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json

# Colab detection
IN_COLAB = ('google.colab' in sys.modules or 
            os.path.exists('/content') or 
            'COLAB_GPU' in os.environ)

if IN_COLAB:
    # Drive should already be mounted from notebook cells
    BASE_DIR = Path('/content/drive/MyDrive/Colab_Data')
else:
    BASE_DIR = Path('.')

# Import required modules
from stable_baselines3 import PPO, DQN
from rl_thin_cloud_environment import ThinCloudDetectionEnv
from rl_thin_cloud_environment_discrete import ThinCloudDetectionEnvDiscrete


def load_test_data():
    """Load test patches from TIF files."""
    print("Loading test data...")
    
    import glob
    import rasterio
    
    # Load from processed TIF files on Drive
    data_dir = '/content/drive/MyDrive/Colab_Data/cloudsen12_processed_1000'
    image_files = sorted(glob.glob(f'{data_dir}/*_image.tif'))
    mask_files = sorted(glob.glob(f'{data_dir}/*_mask.tif'))
    
    # Use test set (last 200 patches, indices 800-999)
    split_idx = int(0.8 * len(image_files))
    test_images = image_files[split_idx:]
    test_masks = mask_files[split_idx:]
    
    print(f"Found {len(test_images)} test images")
    
    patches = []
    labels = []
    
    for img_path, mask_path in zip(test_images, test_masks):
        # Load image
        with rasterio.open(img_path) as src:
            patch = src.read()  # (bands, H, W)
            patch = np.transpose(patch, (1, 2, 0))  # (H, W, bands)
        
        # Load mask
        with rasterio.open(mask_path) as src:
            label = src.read(1)  # (H, W)
        
        patches.append(patch)
        labels.append(label)
    
    return np.array(patches), np.array(labels)


def apply_cnn_baseline(patch):
    """Apply CNN baseline threshold detection using s2cloudless."""
    from s2cloudless import S2PixelCloudDetector
    
    # s2cloudless expects normalized reflectance in [0, 1]
    if patch.max() > 1.0:
        patch_normalized = patch / 10000.0
    else:
        patch_normalized = patch
    
    # Initialize cloud detector
    cloud_detector = S2PixelCloudDetector(threshold=0.4, all_bands=True, average_over=4)
    
    # s2cloudless expects (batch, H, W, bands)
    patch_batched = patch_normalized[np.newaxis, ...]
    cloud_prob = cloud_detector.get_cloud_probability_maps(patch_batched)
    
    # Threshold at 0.5 for binary mask
    cloud_mask = (cloud_prob[0] > 0.5).astype(np.uint8)
    
    return cloud_mask, cloud_prob[0]  # Return both mask and probability


def apply_ppo_model(patch, model, cnn_prob, label):
    """Apply PPO model to get cloud mask."""
    gt_binary = (label > 0).astype(np.uint8)
    env = ThinCloudDetectionEnv(patch, cnn_prob, gt_binary)
    
    # Collect predictions for all patches
    prediction = np.zeros_like(cnn_prob, dtype=np.uint8)
    ps = env.patch_size
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        # Apply action to current patch
        i, j = env.current_pos
        threshold_delta = np.clip(action[0], -0.2, 0.2)
        thin_boost = np.clip(action[1], 0.0, 0.3)
        
        cnn_patch = cnn_prob[i:i+ps, j:j+ps].copy()
        thin_indicator = env.thin_cloud_indicator[i:i+ps, j:j+ps]
        
        boosted_prob = np.clip(cnn_patch + thin_indicator * thin_boost, 0, 1)
        prediction[i:i+ps, j:j+ps] = (boosted_prob > (0.5 + threshold_delta)).astype(np.uint8)
        
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
    
    return prediction


def apply_dqn_model(patch, model, cnn_prob, label):
    """Apply DQN model to get cloud mask."""
    gt_binary = (label > 0).astype(np.uint8)
    env = ThinCloudDetectionEnvDiscrete(patch, cnn_prob, gt_binary)
    
    # Collect predictions for all patches
    prediction = np.zeros_like(cnn_prob, dtype=np.uint8)
    ps = env.patch_size
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        # Decode discrete action
        threshold_idx = action // 3
        boost_idx = action % 3
        threshold_delta = env.THRESHOLD_OPTIONS[threshold_idx]
        thin_boost = env.BOOST_OPTIONS[boost_idx]
        
        # Apply action to current patch
        i, j = env.current_pos
        cnn_patch = cnn_prob[i:i+ps, j:j+ps].copy()
        thin_indicator = env.thin_cloud_indicator[i:i+ps, j:j+ps]
        
        boosted_prob = np.clip(cnn_patch + thin_indicator * thin_boost, 0, 1)
        prediction[i:i+ps, j:j+ps] = (boosted_prob > (0.5 + threshold_delta)).astype(np.uint8)
        
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
    
    return prediction


def create_rgb_image(patch):
    """Create RGB image from patch for visualization."""
    # Use bands 3, 2, 1 (RGB)
    rgb = patch[:, :, [2, 1, 0]]
    
    # Normalize to 0-1 range for display
    rgb = np.clip(rgb / 3000.0, 0, 1)
    
    # Apply gamma correction for better visualization
    rgb = np.power(rgb, 0.7)
    
    return rgb


def create_false_color(patch):
    """Create false color composite (NIR, Red, Green)."""
    # Use bands 7 (NIR), 3 (Red), 2 (Green)
    false_color = patch[:, :, [7, 3, 2]]
    
    # Normalize
    false_color = np.clip(false_color / 3000.0, 0, 1)
    false_color = np.power(false_color, 0.7)
    
    return false_color


def visualize_comparison(patches, labels, cnn_masks, ppo_masks, dqn_masks, 
                        num_samples=5, save_path=None):
    """Create side-by-side comparison visualization with TP/FN/FP overlays."""
    from sklearn.metrics import f1_score
    
    # 7 columns: RGB, Ground Truth, CNN, PPO, DQN, PPO Improvement, DQN Improvement
    fig, axes = plt.subplots(num_samples, 7, figsize=(28, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Prepare ground truth (binary: any cloud = 1)
        gt_binary = (labels[i] > 0).astype(np.uint8)
        thin_cloud_mask = (labels[i] == 2)  # Thin clouds only
        
        # Column 1: RGB Image
        rgb = create_rgb_image(patches[i])
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f'Patch #{i+1}\nRGB Image', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Column 2: Ground Truth with thin clouds highlighted
        gt_display = np.zeros((*labels[i].shape, 3))
        gt_display[:, :, 0] = gt_binary  # All clouds in red
        gt_display[:, :, 1] = thin_cloud_mask  # Thin clouds also in green (makes yellow)
        axes[i, 1].imshow(gt_display)
        thin_pct = thin_cloud_mask.sum() / max(gt_binary.sum(), 1) * 100
        axes[i, 1].set_title(f'Ground Truth\nThin clouds: {thin_pct:.1f}% (yellow)', fontsize=11)
        axes[i, 1].axis('off')
        
        # Column 3: CNN Baseline with TP/FN/FP overlay
        cnn_overlay = np.zeros((*gt_binary.shape, 3))
        cnn_overlay[:, :, 1] = (cnn_masks[i] == 1) & (gt_binary == 1)  # TP green
        cnn_overlay[:, :, 0] = (cnn_masks[i] == 0) & (gt_binary == 1)  # FN red
        cnn_overlay[:, :, 2] = (cnn_masks[i] == 1) & (gt_binary == 0)  # FP blue
        
        axes[i, 2].imshow(cnn_overlay)
        cnn_f1 = f1_score(gt_binary.flatten(), cnn_masks[i].flatten(), zero_division=0)
        cnn_thin_recall = np.sum((cnn_masks[i] == 1) & thin_cloud_mask) / max(thin_cloud_mask.sum(), 1)
        axes[i, 2].set_title(f'CNN Baseline\nThin Recall: {cnn_thin_recall*100:.1f}%\nF1: {cnn_f1:.3f}', fontsize=11)
        axes[i, 2].axis('off')
        
        # Column 4: PPO with TP/FN/FP overlay
        ppo_overlay = np.zeros((*gt_binary.shape, 3))
        ppo_overlay[:, :, 1] = (ppo_masks[i] == 1) & (gt_binary == 1)  # TP green
        ppo_overlay[:, :, 0] = (ppo_masks[i] == 0) & (gt_binary == 1)  # FN red
        ppo_overlay[:, :, 2] = (ppo_masks[i] == 1) & (gt_binary == 0)  # FP blue
        
        axes[i, 3].imshow(ppo_overlay)
        ppo_f1 = f1_score(gt_binary.flatten(), ppo_masks[i].flatten(), zero_division=0)
        ppo_thin_recall = np.sum((ppo_masks[i] == 1) & thin_cloud_mask) / max(thin_cloud_mask.sum(), 1)
        axes[i, 3].set_title(f'PPO\nThin Recall: {ppo_thin_recall*100:.1f}%\nF1: {ppo_f1:.3f}', fontsize=11)
        axes[i, 3].axis('off')
        
        # Column 5: DQN with TP/FN/FP overlay
        dqn_overlay = np.zeros((*gt_binary.shape, 3))
        dqn_overlay[:, :, 1] = (dqn_masks[i] == 1) & (gt_binary == 1)  # TP green
        dqn_overlay[:, :, 0] = (dqn_masks[i] == 0) & (gt_binary == 1)  # FN red
        dqn_overlay[:, :, 2] = (dqn_masks[i] == 1) & (gt_binary == 0)  # FP blue
        
        axes[i, 4].imshow(dqn_overlay)
        dqn_f1 = f1_score(gt_binary.flatten(), dqn_masks[i].flatten(), zero_division=0)
        dqn_thin_recall = np.sum((dqn_masks[i] == 1) & thin_cloud_mask) / max(thin_cloud_mask.sum(), 1)
        axes[i, 4].set_title(f'DQN\nThin Recall: {dqn_thin_recall*100:.1f}%\nF1: {dqn_f1:.3f}', fontsize=11)
        axes[i, 4].axis('off')
        
        # Column 6: PPO Improvement (Green=Fixed, Red=Lost, Cyan=Thin improved)
        ppo_improvement = np.zeros((*gt_binary.shape, 3))
        # Green: PPO fixed (baseline missed, PPO caught)
        ppo_improvement[:, :, 1] = (ppo_masks[i] == 1) & (cnn_masks[i] == 0) & (gt_binary == 1)
        # Red: PPO lost (baseline caught, PPO missed)
        ppo_improvement[:, :, 0] = (ppo_masks[i] == 0) & (cnn_masks[i] == 1) & (gt_binary == 1)
        # Cyan: Thin clouds that PPO improved
        ppo_thin_improved = thin_cloud_mask & (ppo_masks[i] == 1) & (cnn_masks[i] == 0)
        ppo_improvement[:, :, 2] = np.maximum(ppo_improvement[:, :, 2], ppo_thin_improved.astype(float))
        ppo_improvement[:, :, 1] = np.maximum(ppo_improvement[:, :, 1], ppo_thin_improved.astype(float))
        
        axes[i, 5].imshow(ppo_improvement)
        ppo_thin_improvement = ppo_thin_recall - cnn_thin_recall
        axes[i, 5].set_title(f'PPO Improvement\n+{ppo_thin_improvement*100:.1f}% thin recall\nGreen=Fixed, Red=Lost, Cyan=Thin', fontsize=10)
        axes[i, 5].axis('off')
        
        # Column 7: DQN Improvement (Green=Fixed, Red=Lost, Cyan=Thin improved)
        dqn_improvement = np.zeros((*gt_binary.shape, 3))
        # Green: DQN fixed (baseline missed, DQN caught)
        dqn_improvement[:, :, 1] = (dqn_masks[i] == 1) & (cnn_masks[i] == 0) & (gt_binary == 1)
        # Red: DQN lost (baseline caught, DQN missed)
        dqn_improvement[:, :, 0] = (dqn_masks[i] == 0) & (cnn_masks[i] == 1) & (gt_binary == 1)
        # Cyan: Thin clouds that DQN improved
        dqn_thin_improved = thin_cloud_mask & (dqn_masks[i] == 1) & (cnn_masks[i] == 0)
        dqn_improvement[:, :, 2] = np.maximum(dqn_improvement[:, :, 2], dqn_thin_improved.astype(float))
        dqn_improvement[:, :, 1] = np.maximum(dqn_improvement[:, :, 1], dqn_thin_improved.astype(float))
        
        axes[i, 6].imshow(dqn_improvement)
        dqn_thin_improvement = dqn_thin_recall - cnn_thin_recall
        axes[i, 6].set_title(f'DQN Improvement\n+{dqn_thin_improvement*100:.1f}% thin recall\nGreen=Fixed, Red=Lost, Cyan=Thin', fontsize=10)
        axes[i, 6].axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=[0, 1, 0], label='True Positive (TP) / Fixed'),
        mpatches.Patch(color=[1, 0, 0], label='False Negative (FN) / Lost'),
        mpatches.Patch(color=[0, 0, 1], label='False Positive (FP)'),
        mpatches.Patch(color=[1, 1, 0], label='Thin Cloud (GT)'),
        mpatches.Patch(color=[0, 1, 1], label='Thin Cloud Improved'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 0.02), ncol=5, fontsize=10)
    
    plt.suptitle('Algorithm Comparison: CNN Baseline vs PPO vs DQN\n(Columns 1-5: Green=TP, Red=FN, Blue=FP | Columns 6-7: Green=Fixed, Red=Lost, Cyan=Thin Improved)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    
    plt.show()


def main():
    print("="*60)
    print("Algorithm Mask Comparison Visualization")
    print("="*60)
    
    # Load test data
    patches, labels = load_test_data()
    print(f"Loaded {len(patches)} test patches")
    
    # Load PPO model
    print("\nLoading PPO model...")
    ppo_model_path = BASE_DIR / 'thin_cloud_v2' / 'thin_cloud_720000_steps.zip'
    if not ppo_model_path.exists():
        print(f"ERROR: PPO model not found at {ppo_model_path}")
        return
    ppo_model = PPO.load(str(ppo_model_path))
    print("PPO model loaded successfully")
    
    # Load DQN model
    print("Loading DQN model...")
    dqn_model_path = BASE_DIR / 'dqn_thin_cloud' / 'dqn_thin_cloud_100000_steps.zip'
    if not dqn_model_path.exists():
        print(f"ERROR: DQN model not found at {dqn_model_path}")
        return
    dqn_model = DQN.load(str(dqn_model_path))
    print("DQN model loaded successfully")
    
    # Select samples with BEST thin cloud improvement (like original PPO visualization)
    print("\nFinding patches with best thin cloud improvement...")
    
    samples_with_improvement = []
    
    for idx in range(min(50, len(patches))):  # Check first 50 patches
        patch = patches[idx]
        label = labels[idx]
        
        # Get thin cloud mask (label == 2)
        thin_cloud_mask = (label == 2)
        thin_cloud_count = thin_cloud_mask.sum()
        
        # Skip patches with too few thin clouds
        if thin_cloud_count < 100:
            continue
        
        # Get predictions
        cnn_mask, cnn_prob = apply_cnn_baseline(patch)
        dqn_mask = apply_dqn_model(patch, dqn_model, cnn_prob, label)
        ppo_mask = apply_ppo_model(patch, ppo_model, cnn_prob, label)
        
        # Calculate thin cloud recall for each
        cnn_thin_recall = np.sum((cnn_mask == 1) & thin_cloud_mask) / thin_cloud_count
        dqn_thin_recall = np.sum((dqn_mask == 1) & thin_cloud_mask) / thin_cloud_count
        ppo_thin_recall = np.sum((ppo_mask == 1) & thin_cloud_mask) / thin_cloud_count
        
        # Calculate improvement (best of DQN or PPO over CNN)
        dqn_improvement = dqn_thin_recall - cnn_thin_recall
        ppo_improvement = ppo_thin_recall - cnn_thin_recall
        best_improvement = max(dqn_improvement, ppo_improvement)
        
        if best_improvement > 0.05:  # At least 5% improvement
            samples_with_improvement.append({
                'idx': idx,
                'patch': patch,
                'label': label,
                'cnn_mask': cnn_mask,
                'cnn_prob': cnn_prob,
                'ppo_mask': ppo_mask,
                'dqn_mask': dqn_mask,
                'cnn_thin_recall': cnn_thin_recall,
                'ppo_thin_recall': ppo_thin_recall,
                'dqn_thin_recall': dqn_thin_recall,
                'dqn_improvement': dqn_improvement,
                'ppo_improvement': ppo_improvement,
                'best_improvement': best_improvement,
                'thin_cloud_count': thin_cloud_count
            })
            print(f"  Patch {idx}: CNN={cnn_thin_recall*100:.1f}% → DQN={dqn_thin_recall*100:.1f}% (+{dqn_improvement*100:.1f}%), PPO={ppo_thin_recall*100:.1f}% (+{ppo_improvement*100:.1f}%)")
    
    # Sort by best improvement and take top 5
    samples_with_improvement.sort(key=lambda x: x['best_improvement'], reverse=True)
    selected_samples = samples_with_improvement[:5]
    
    if len(selected_samples) == 0:
        print("ERROR: No patches with thin cloud improvement found!")
        return
    
    print(f"\n✅ Selected {len(selected_samples)} best patches for visualization")
    
    # Extract data for visualization
    selected_indices = [s['idx'] for s in selected_samples]
    selected_patches = [s['patch'] for s in selected_samples]
    selected_labels = [s['label'] for s in selected_samples]
    
    # Use pre-computed masks from sample selection
    cnn_masks = [s['cnn_mask'] for s in selected_samples]
    ppo_masks = [s['ppo_mask'] for s in selected_samples]
    dqn_masks = [s['dqn_mask'] for s in selected_samples]
    
    print(f"Using pre-computed predictions for {len(selected_samples)} patches")
    
    # Create visualization
    print("\nCreating visualization...")
    save_path = None
    if IN_COLAB:
        save_path = BASE_DIR / 'algorithm_comparison' / 'algorithm_mask_comparison.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        save_path = Path('results') / 'algorithm_mask_comparison.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    visualize_comparison(
        selected_patches, 
        selected_labels, 
        cnn_masks, 
        ppo_masks, 
        dqn_masks,
        num_samples=5,
        save_path=str(save_path)
    )
    
    # Calculate and display metrics for selected patches
    print("\n" + "="*60)
    print("Metrics for Selected Patches:")
    print("="*60)
    
    for i, sample in enumerate(selected_samples):
        print(f"\nPatch {i+1} (Index {sample['idx']}):")
        
        thin_cloud_mask = (sample['label'] == 2)
        thin_cloud_pct = np.sum(thin_cloud_mask) / sample['label'].size * 100
        
        # Display metrics from pre-computed values
        print(f"  Thin Cloud Coverage: {thin_cloud_pct:.1f}%")
        print(f"  Thin Cloud Recall:")
        print(f"    CNN: {sample['cnn_thin_recall']*100:.1f}%")
        print(f"    PPO: {sample['ppo_thin_recall']*100:.1f}% (+{sample['ppo_improvement']*100:.1f}%)")
        print(f"    DQN: {sample['dqn_thin_recall']*100:.1f}% (+{sample['dqn_improvement']*100:.1f}%)")
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == '__main__':
    main()
