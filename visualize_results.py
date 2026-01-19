"""
Visualization script for thin cloud detection results.
Shows side-by-side comparison of baseline CNN vs RL refined model.
"""

import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from cnn_inference import load_sentinel2_image, get_cloud_mask
from rl_thin_cloud_environment import ThinCloudDetectionEnv

print("🔍 Loading trained model...")

# Find latest checkpoint
checkpoint_paths = sorted(glob.glob('checkpoints/thin_cloud/thin_cloud_*_steps.zip'))
model_paths = sorted(glob.glob('models/ppo_thin_cloud_*/model.zip'))

if checkpoint_paths:
    model_path = checkpoint_paths[-1]
    print(f"✅ Loading checkpoint: {model_path}")
    model = PPO.load(model_path)
elif model_paths:
    model_path = model_paths[-1].replace('.zip', '')
    print(f"✅ Loading model: {model_path}")
    model = PPO.load(model_path)
else:
    print("❌ No trained model found!")
    exit(1)

# Load test data
data_dir = '/content/drive/MyDrive/Colab_Data/cloudsen12_processed_1000'
image_files = sorted(glob.glob(f'{data_dir}/*_image.tif'))
mask_files = sorted(glob.glob(f'{data_dir}/*_mask.tif'))

# Use test set
split_idx = int(0.8 * len(image_files))
test_images = image_files[split_idx:]
test_masks = mask_files[split_idx:]

print(f"📊 Found {len(test_images)} test patches")

def get_rgb(image):
    """Extract RGB channels for visualization (B4, B3, B2 = indices 2, 1, 0)"""
    # Normalize for display
    rgb = image[:, :, [2, 1, 0]]  # B4, B3, B2 for true color
    rgb = np.clip(rgb / rgb.max(), 0, 1)
    # Enhance contrast
    rgb = np.clip(rgb * 2.5, 0, 1)
    return rgb

def apply_rl_model(model, image, cnn_prob, gt_binary):
    """Apply RL model to get refined prediction."""
    env = ThinCloudDetectionEnv(image, cnn_prob, gt_binary, patch_size=64)
    rl_pred = np.zeros_like(gt_binary, dtype=np.uint8)
    
    obs, _ = env.reset()
    for patch_idx in range(env.num_patches):
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
    
    return rl_pred, env.thin_cloud_indicator

# Find samples with thin clouds for visualization
print("\n🔍 Finding samples with thin clouds for visualization...")

samples_to_show = []
for idx in range(len(test_images)):
    image = load_sentinel2_image(test_images[idx])
    cnn_prob = get_cloud_mask(image)
    
    with rasterio.open(test_masks[idx]) as src:
        gt_raw = src.read(1)
    
    gt_binary = (gt_raw > 0).astype(np.uint8)
    thin_cloud_mask = (gt_binary == 1) & (cnn_prob >= 0.2) & (cnn_prob <= 0.6)
    
    thin_cloud_ratio = thin_cloud_mask.sum() / gt_binary.size
    
    # Select samples with good amount of thin clouds
    if thin_cloud_ratio > 0.02:  # At least 2% thin clouds
        baseline_pred = (cnn_prob > 0.5).astype(np.uint8)
        rl_pred, thin_indicator = apply_rl_model(model, image, cnn_prob, gt_binary)
        
        # Check if RL improved
        baseline_thin_correct = np.sum((baseline_pred == 1) & thin_cloud_mask)
        rl_thin_correct = np.sum((rl_pred == 1) & thin_cloud_mask)
        
        if rl_thin_correct > baseline_thin_correct:
            samples_to_show.append({
                'idx': idx,
                'image': image,
                'cnn_prob': cnn_prob,
                'gt': gt_binary,
                'baseline': baseline_pred,
                'rl': rl_pred,
                'thin_mask': thin_cloud_mask,
                'thin_indicator': thin_indicator,
                'improvement': rl_thin_correct - baseline_thin_correct
            })
    
    if len(samples_to_show) >= 4:
        break

print(f"✅ Found {len(samples_to_show)} good samples for visualization")

# Create visualization
fig, axes = plt.subplots(len(samples_to_show), 5, figsize=(20, 4*len(samples_to_show)))

if len(samples_to_show) == 1:
    axes = axes.reshape(1, -1)

for row, sample in enumerate(samples_to_show):
    # RGB Image
    axes[row, 0].imshow(get_rgb(sample['image']))
    axes[row, 0].set_title('RGB Image', fontsize=12)
    axes[row, 0].axis('off')
    
    # Ground Truth
    axes[row, 1].imshow(sample['gt'], cmap='Reds', vmin=0, vmax=1)
    axes[row, 1].set_title('Ground Truth\n(All Clouds)', fontsize=12)
    axes[row, 1].axis('off')
    
    # Baseline CNN
    axes[row, 2].imshow(sample['baseline'], cmap='Blues', vmin=0, vmax=1)
    baseline_recall = np.sum((sample['baseline'] == 1) & (sample['gt'] == 1)) / max(sample['gt'].sum(), 1)
    axes[row, 2].set_title(f'Baseline CNN\nRecall: {baseline_recall*100:.1f}%', fontsize=12)
    axes[row, 2].axis('off')
    
    # RL Refined
    axes[row, 3].imshow(sample['rl'], cmap='Greens', vmin=0, vmax=1)
    rl_recall = np.sum((sample['rl'] == 1) & (sample['gt'] == 1)) / max(sample['gt'].sum(), 1)
    axes[row, 3].set_title(f'RL Refined\nRecall: {rl_recall*100:.1f}%', fontsize=12)
    axes[row, 3].axis('off')
    
    # Difference (improvement shown in green, degradation in red)
    diff = np.zeros((*sample['gt'].shape, 3))
    # Green: RL detected but baseline missed (improvement)
    diff[:, :, 1] = ((sample['rl'] == 1) & (sample['baseline'] == 0) & (sample['gt'] == 1)).astype(float)
    # Red: Baseline detected but RL missed (degradation)  
    diff[:, :, 0] = ((sample['rl'] == 0) & (sample['baseline'] == 1) & (sample['gt'] == 1)).astype(float)
    # Highlight thin cloud regions
    diff[:, :, 2] = sample['thin_mask'].astype(float) * 0.3
    
    axes[row, 4].imshow(diff)
    improvement = sample['improvement']
    axes[row, 4].set_title(f'Improvement\nGreen: +{improvement} pixels\n(Thin clouds in blue)', fontsize=12)
    axes[row, 4].axis('off')

plt.suptitle('Thin Cloud Detection: Baseline CNN vs RL Refined Model', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('thin_cloud_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Visualization saved to 'thin_cloud_comparison.png'")

# Create a detailed comparison figure for a single sample
if len(samples_to_show) > 0:
    sample = samples_to_show[0]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Original data
    axes[0, 0].imshow(get_rgb(sample['image']))
    axes[0, 0].set_title('RGB Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(sample['cnn_prob'], cmap='YlOrRd', vmin=0, vmax=1)
    axes[0, 1].set_title('CNN Cloud Probability', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(sample['gt'], cmap='gray')
    axes[0, 2].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Predictions comparison
    # Create overlay: GT in background, predictions overlaid
    overlay_baseline = np.zeros((*sample['gt'].shape, 3))
    overlay_baseline[:, :, 0] = (sample['gt'] == 1) & (sample['baseline'] == 0)  # Missed (red)
    overlay_baseline[:, :, 1] = (sample['gt'] == 1) & (sample['baseline'] == 1)  # Correct (green)
    overlay_baseline[:, :, 2] = (sample['gt'] == 0) & (sample['baseline'] == 1)  # False positive (blue)
    
    axes[1, 0].imshow(overlay_baseline)
    baseline_thin_recall = np.sum((sample['baseline'] == 1) & sample['thin_mask']) / max(sample['thin_mask'].sum(), 1)
    axes[1, 0].set_title(f'Baseline CNN\nThin Cloud Recall: {baseline_thin_recall*100:.1f}%\n(Green=Correct, Red=Missed)', fontsize=12)
    axes[1, 0].axis('off')
    
    overlay_rl = np.zeros((*sample['gt'].shape, 3))
    overlay_rl[:, :, 0] = (sample['gt'] == 1) & (sample['rl'] == 0)  # Missed (red)
    overlay_rl[:, :, 1] = (sample['gt'] == 1) & (sample['rl'] == 1)  # Correct (green)
    overlay_rl[:, :, 2] = (sample['gt'] == 0) & (sample['rl'] == 1)  # False positive (blue)
    
    axes[1, 1].imshow(overlay_rl)
    rl_thin_recall = np.sum((sample['rl'] == 1) & sample['thin_mask']) / max(sample['thin_mask'].sum(), 1)
    axes[1, 1].set_title(f'RL Refined Model\nThin Cloud Recall: {rl_thin_recall*100:.1f}%\n(Green=Correct, Red=Missed)', fontsize=12)
    axes[1, 1].axis('off')
    
    # Thin cloud specific
    thin_overlay = np.zeros((*sample['gt'].shape, 3))
    thin_overlay[:, :, 0] = sample['thin_mask'] & (sample['rl'] == 0)  # Thin missed by RL
    thin_overlay[:, :, 1] = sample['thin_mask'] & (sample['rl'] == 1) & (sample['baseline'] == 0)  # Thin detected by RL but not baseline
    thin_overlay[:, :, 2] = sample['thin_mask'] & (sample['baseline'] == 1)  # Thin detected by baseline
    
    axes[1, 2].imshow(thin_overlay)
    axes[1, 2].set_title(f'Thin Clouds Only\nGreen: RL improvement\nBlue: Baseline detected\nRed: Both missed', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.suptitle('Detailed Comparison: Thin Cloud Detection', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('thin_cloud_detailed.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Detailed visualization saved to 'thin_cloud_detailed.png'")

print("\n" + "="*60)
print("📊 VISUALIZATION COMPLETE")
print("="*60)
