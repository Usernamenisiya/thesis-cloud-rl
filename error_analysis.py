"""
Error Analysis: Identify and visualize failure cases.
Analyze where RL helps most and where it still struggles.
"""

import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from stable_baselines3 import PPO
from cnn_inference import load_sentinel2_image, get_cloud_mask
from rl_thin_cloud_environment import ThinCloudDetectionEnv

print("🔍 ERROR ANALYSIS")
print("="*70)
print("Identifying and categorizing failure cases")
print("="*70)

# Load model - check multiple possible locations
checkpoint_locations = [
    '/content/drive/MyDrive/Colab_Data/checkpoints/thin_cloud/thin_cloud_*_steps.zip',
    'checkpoints/thin_cloud/thin_cloud_*_steps.zip',
    '/content/drive/MyDrive/checkpoints/thin_cloud/thin_cloud_*_steps.zip'
]

checkpoint_paths = []
for loc in checkpoint_locations:
    checkpoint_paths = sorted(glob.glob(loc))
    if checkpoint_paths:
        break

if not checkpoint_paths:
    print("❌ No checkpoints found! Tried:")
    for loc in checkpoint_locations:
        print(f"   - {loc}")
    exit(1)

model_path = checkpoint_paths[-1]
print(f"\n✅ Loading model: {model_path}")
model = PPO.load(model_path)

# Load test data
data_dir = '/content/drive/MyDrive/Colab_Data/cloudsen12_processed_1000'
image_files = sorted(glob.glob(f'{data_dir}/*_image.tif'))
mask_files = sorted(glob.glob(f'{data_dir}/*_mask.tif'))

split_idx = int(0.8 * len(image_files))
test_images = image_files[split_idx:]
test_masks = mask_files[split_idx:]

# Categories for analysis
failure_cases = {
    'baseline_better': [],  # RL makes it worse
    'both_fail': [],  # Both miss thin clouds
    'both_succeed': [],  # Both catch thin clouds
    'rl_fixes': []  # RL successfully fixes baseline error
}

num_samples = min(len(test_images), 50)  # Analyze 50 patches in detail

for idx in range(num_samples):
    image = load_sentinel2_image(test_images[idx])
    cnn_prob = get_cloud_mask(image)
    
    with rasterio.open(test_masks[idx]) as src:
        gt_raw = src.read(1)
    
    gt_binary = (gt_raw > 0).astype(np.uint8)
    baseline_pred = (cnn_prob > 0.5).astype(np.uint8)
    
    # Thin cloud mask
    thin_cloud_mask = (gt_binary == 1) & (cnn_prob >= 0.2) & (cnn_prob <= 0.6)
    
    if thin_cloud_mask.sum() == 0:
        continue  # Skip patches without thin clouds
    
    # Get RL prediction
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
    
    # Compute thin cloud recall for this patch
    baseline_correct = np.sum((baseline_pred == 1) & thin_cloud_mask)
    rl_correct = np.sum((rl_pred == 1) & thin_cloud_mask)
    total_thin = thin_cloud_mask.sum()
    
    baseline_recall = baseline_correct / total_thin
    rl_recall = rl_correct / total_thin
    
    case_data = {
        'idx': idx,
        'filename': test_images[idx].split('/')[-1],
        'baseline_recall': baseline_recall,
        'rl_recall': rl_recall,
        'improvement': rl_recall - baseline_recall,
        'thin_pixels': total_thin,
        'image': image,
        'cnn_prob': cnn_prob,
        'gt_binary': gt_binary,
        'thin_mask': thin_cloud_mask,
        'baseline_pred': baseline_pred,
        'rl_pred': rl_pred
    }
    
    # Categorize
    if rl_recall < baseline_recall - 0.05:  # RL significantly worse
        failure_cases['baseline_better'].append(case_data)
    elif baseline_recall < 0.3 and rl_recall < 0.3:  # Both fail
        failure_cases['both_fail'].append(case_data)
    elif baseline_recall > 0.7 and rl_recall > 0.7:  # Both succeed
        failure_cases['both_succeed'].append(case_data)
    elif rl_recall > baseline_recall + 0.1:  # RL significantly better
        failure_cases['rl_fixes'].append(case_data)
    
    if (idx + 1) % 10 == 0:
        print(f"  Analyzed {idx + 1}/{num_samples} patches...")

# Print summary
print("\n" + "="*70)
print("📊 ERROR ANALYSIS SUMMARY")
print("="*70)

total_analyzed = sum(len(cases) for cases in failure_cases.values())
print(f"\n✅ Analyzed {total_analyzed} patches with thin clouds\n")

for category, cases in failure_cases.items():
    pct = len(cases) / total_analyzed * 100 if total_analyzed > 0 else 0
    print(f"{category.upper().replace('_', ' '):<25} {len(cases):>4} patches ({pct:>5.1f}%)")

# Visualize examples from each category
fig, axes = plt.subplots(4, 3, figsize=(15, 20))
categories = ['baseline_better', 'both_fail', 'rl_fixes', 'both_succeed']
titles = ['RL Makes It Worse', 'Both Methods Fail', 'RL Successfully Fixes', 'Both Methods Succeed']

for row, (category, title) in enumerate(zip(categories, titles)):
    cases = failure_cases[category]
    
    if len(cases) == 0:
        for col in range(3):
            axes[row, col].text(0.5, 0.5, 'No cases', ha='center', va='center')
            axes[row, col].set_title(f'{title}\n(No examples)')
            axes[row, col].axis('off')
        continue
    
    # Show worst case (or best improvement for rl_fixes)
    if category == 'rl_fixes':
        case = max(cases, key=lambda x: x['improvement'])
    else:
        case = cases[0] if len(cases) > 0 else None
    
    if case is None:
        continue
    
    # RGB visualization
    rgb = case['image'][:3].transpose(1, 2, 0)
    rgb = np.clip(rgb / 3000, 0, 1)  # Normalize for display
    
    # Baseline
    ax = axes[row, 0]
    ax.imshow(rgb)
    # Overlay errors
    baseline_errors = case['thin_mask'] & (case['baseline_pred'] == 0)
    if baseline_errors.sum() > 0:
        error_overlay = np.zeros((*baseline_errors.shape, 4))
        error_overlay[baseline_errors] = [1, 0, 0, 0.5]  # Red for missed
        ax.imshow(error_overlay)
    ax.set_title(f'Baseline\nRecall: {case["baseline_recall"]*100:.1f}%')
    ax.axis('off')
    
    # RL
    ax = axes[row, 1]
    ax.imshow(rgb)
    rl_errors = case['thin_mask'] & (case['rl_pred'] == 0)
    if rl_errors.sum() > 0:
        error_overlay = np.zeros((*rl_errors.shape, 4))
        error_overlay[rl_errors] = [1, 0, 0, 0.5]
        ax.imshow(error_overlay)
    ax.set_title(f'RL Model\nRecall: {case["rl_recall"]*100:.1f}%')
    ax.axis('off')
    
    # Difference
    ax = axes[row, 2]
    diff = np.zeros((*case['gt_binary'].shape, 3))
    
    # Green: RL fixed (RL correct, baseline wrong)
    rl_fixes_mask = case['thin_mask'] & (case['rl_pred'] == 1) & (case['baseline_pred'] == 0)
    diff[rl_fixes_mask] = [0, 1, 0]
    
    # Red: RL broke (baseline correct, RL wrong)
    rl_breaks_mask = case['thin_mask'] & (case['baseline_pred'] == 1) & (case['rl_pred'] == 0)
    diff[rl_breaks_mask] = [1, 0, 0]
    
    # Gray: Both correct
    both_correct = case['thin_mask'] & (case['baseline_pred'] == 1) & (case['rl_pred'] == 1)
    diff[both_correct] = [0.5, 0.5, 0.5]
    
    ax.imshow(diff)
    ax.set_title(f'Difference\nΔ: {case["improvement"]*100:+.1f}%\nGreen=Fixed, Red=Broke')
    ax.axis('off')

plt.tight_layout()
plt.savefig('error_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Saved visualization: error_analysis.png")

# Additional statistics
print("\n" + "="*70)
print("📈 DETAILED STATISTICS BY CATEGORY")
print("="*70)

for category, title in zip(categories, titles):
    cases = failure_cases[category]
    if len(cases) == 0:
        continue
    
    avg_baseline = np.mean([c['baseline_recall'] for c in cases])
    avg_rl = np.mean([c['rl_recall'] for c in cases])
    avg_improvement = np.mean([c['improvement'] for c in cases])
    avg_thin_pixels = np.mean([c['thin_pixels'] for c in cases])
    
    print(f"\n{title.upper()}:")
    print(f"  Count: {len(cases)} patches")
    print(f"  Avg baseline recall: {avg_baseline*100:.2f}%")
    print(f"  Avg RL recall: {avg_rl*100:.2f}%")
    print(f"  Avg improvement: {avg_improvement*100:+.2f}%")
    print(f"  Avg thin cloud pixels: {avg_thin_pixels:.0f}")

# Identify common characteristics of failures
print("\n" + "="*70)
print("💡 KEY INSIGHTS")
print("="*70)

if len(failure_cases['both_fail']) > 0:
    avg_pixels = np.mean([c['thin_pixels'] for c in failure_cases['both_fail']])
    print(f"\n⚠️  Patches where both fail have {avg_pixels:.0f} avg thin cloud pixels")

if len(failure_cases['rl_fixes']) > 0:
    best_case = max(failure_cases['rl_fixes'], key=lambda x: x['improvement'])
    print(f"\n✅ Best RL improvement: {best_case['improvement']*100:+.2f}% on {best_case['filename']}")

if len(failure_cases['baseline_better']) > 0:
    worst_case = min(failure_cases['baseline_better'], key=lambda x: x['improvement'])
    print(f"\n⚠️  Worst RL degradation: {worst_case['improvement']*100:.2f}% on {worst_case['filename']}")

print("\n" + "="*70)
print("✅ Error analysis complete!")
print("="*70)
