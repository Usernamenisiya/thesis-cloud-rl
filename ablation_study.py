"""
Ablation Study: Test model without thin cloud boost action.
This shows the importance of the thin cloud boost feature.

Run this after restarting Colab and loading the model.
"""

import glob
import numpy as np
import rasterio
from stable_baselines3 import PPO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from cnn_inference import load_sentinel2_image, get_cloud_mask
from rl_thin_cloud_environment import ThinCloudDetectionEnv

print("🔬 ABLATION STUDY: Thin Cloud Boost")
print("="*70)
print("Testing model performance with and without thin cloud boost action")
print("="*70)

# Load model - check multiple possible locations
checkpoint_locations = [
    '/content/drive/MyDrive/Colab_Data/checkpoints/thin_cloud/thin_cloud_*_steps.zip',
    '/content/drive/MyDrive/Colab_Data/models/ppo_thin_cloud_*/model.zip',
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

print(f"📊 Evaluating on {len(test_images)} test patches")

# Metrics for three variants
baseline_metrics = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
full_rl_metrics = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
no_boost_metrics = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

# Thin cloud specific
thin_baseline = {'tp': 0, 'fn': 0}
thin_full_rl = {'tp': 0, 'fn': 0}
thin_no_boost = {'tp': 0, 'fn': 0}

num_samples = min(len(test_images), 100)  # Use 100 for speed

for idx in range(num_samples):
    image = load_sentinel2_image(test_images[idx])
    cnn_prob = get_cloud_mask(image)
    
    with rasterio.open(test_masks[idx]) as src:
        gt_raw = src.read(1)
    
    gt_binary = (gt_raw > 0).astype(np.uint8)
    baseline_pred = (cnn_prob > 0.5).astype(np.uint8)
    
    # Thin cloud mask
    thin_cloud_mask = (gt_binary == 1) & (cnn_prob >= 0.2) & (cnn_prob <= 0.6)
    
    # Create environment
    env = ThinCloudDetectionEnv(image, cnn_prob, gt_binary, patch_size=64)
    
    # Full RL model (with thin cloud boost)
    full_rl_pred = np.zeros_like(gt_binary, dtype=np.uint8)
    # No boost variant (threshold adjustment only)
    no_boost_pred = np.zeros_like(gt_binary, dtype=np.uint8)
    
    obs, _ = env.reset()
    for patch_idx in range(env.num_patches):
        action, _ = model.predict(obs, deterministic=True)
        
        i, j = env.current_pos
        ps = env.patch_size
        
        threshold_delta = np.clip(action[0], -0.2, 0.2)
        thin_boost = np.clip(action[1], 0.0, 0.3)
        
        cnn_patch = cnn_prob[i:i+ps, j:j+ps].copy()
        thin_indicator = env.thin_cloud_indicator[i:i+ps, j:j+ps]
        
        # Full model: threshold + boost
        boosted_prob = np.clip(cnn_patch + thin_indicator * thin_boost, 0, 1)
        full_rl_pred[i:i+ps, j:j+ps] = (boosted_prob > (0.5 + threshold_delta)).astype(np.uint8)
        
        # Ablation: threshold only, NO boost
        no_boost_pred[i:i+ps, j:j+ps] = (cnn_patch > (0.5 + threshold_delta)).astype(np.uint8)
        
        obs, _, done, _, _ = env.step(action)
        if done:
            break
    
    # Accumulate metrics - Baseline
    baseline_metrics['tp'] += np.sum((baseline_pred == 1) & (gt_binary == 1))
    baseline_metrics['fp'] += np.sum((baseline_pred == 1) & (gt_binary == 0))
    baseline_metrics['tn'] += np.sum((baseline_pred == 0) & (gt_binary == 0))
    baseline_metrics['fn'] += np.sum((baseline_pred == 0) & (gt_binary == 1))
    
    # Full RL
    full_rl_metrics['tp'] += np.sum((full_rl_pred == 1) & (gt_binary == 1))
    full_rl_metrics['fp'] += np.sum((full_rl_pred == 1) & (gt_binary == 0))
    full_rl_metrics['tn'] += np.sum((full_rl_pred == 0) & (gt_binary == 0))
    full_rl_metrics['fn'] += np.sum((full_rl_pred == 0) & (gt_binary == 1))
    
    # No boost
    no_boost_metrics['tp'] += np.sum((no_boost_pred == 1) & (gt_binary == 1))
    no_boost_metrics['fp'] += np.sum((no_boost_pred == 1) & (gt_binary == 0))
    no_boost_metrics['tn'] += np.sum((no_boost_pred == 0) & (gt_binary == 0))
    no_boost_metrics['fn'] += np.sum((no_boost_pred == 0) & (gt_binary == 1))
    
    # Thin cloud specific
    if thin_cloud_mask.sum() > 0:
        thin_baseline['tp'] += np.sum((baseline_pred == 1) & thin_cloud_mask)
        thin_baseline['fn'] += np.sum((baseline_pred == 0) & thin_cloud_mask)
        
        thin_full_rl['tp'] += np.sum((full_rl_pred == 1) & thin_cloud_mask)
        thin_full_rl['fn'] += np.sum((full_rl_pred == 0) & thin_cloud_mask)
        
        thin_no_boost['tp'] += np.sum((no_boost_pred == 1) & thin_cloud_mask)
        thin_no_boost['fn'] += np.sum((no_boost_pred == 0) & thin_cloud_mask)
    
    if (idx + 1) % 20 == 0:
        print(f"  Processed {idx + 1}/{num_samples} patches...")

# Compute metrics
def compute_metrics(m):
    tp, fp, tn, fn = m['tp'], m['fp'], m['tn'], m['fn']
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    return accuracy, precision, recall, f1, iou

print("\n" + "="*70)
print("📊 ABLATION STUDY RESULTS")
print("="*70)

# Overall metrics
acc_base, prec_base, rec_base, f1_base, iou_base = compute_metrics(baseline_metrics)
acc_full, prec_full, rec_full, f1_full, iou_full = compute_metrics(full_rl_metrics)
acc_no, prec_no, rec_no, f1_no, iou_no = compute_metrics(no_boost_metrics)

print("\n📈 OVERALL CLOUD DETECTION:")
print(f"\n{'Variant':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'IoU':<12}")
print("-"*80)
print(f"{'Baseline CNN':<30} {acc_base*100:>10.2f}% {prec_base*100:>10.2f}% {rec_base*100:>10.2f}% {f1_base*100:>10.2f}% {iou_base:>10.4f}")
print(f"{'RL: Threshold Only (No Boost)':<30} {acc_no*100:>10.2f}% {prec_no*100:>10.2f}% {rec_no*100:>10.2f}% {f1_no*100:>10.2f}% {iou_no:>10.4f}")
print(f"{'RL: Full Model (With Boost)':<30} {acc_full*100:>10.2f}% {prec_full*100:>10.2f}% {rec_full*100:>10.2f}% {f1_full*100:>10.2f}% {iou_full:>10.4f}")

# Thin cloud specific
thin_rec_base = thin_baseline['tp'] / (thin_baseline['tp'] + thin_baseline['fn']) if (thin_baseline['tp'] + thin_baseline['fn']) > 0 else 0
thin_rec_no = thin_no_boost['tp'] / (thin_no_boost['tp'] + thin_no_boost['fn']) if (thin_no_boost['tp'] + thin_no_boost['fn']) > 0 else 0
thin_rec_full = thin_full_rl['tp'] / (thin_full_rl['tp'] + thin_full_rl['fn']) if (thin_full_rl['tp'] + thin_full_rl['fn']) > 0 else 0

print("\n" + "="*70)
print("🎯 THIN CLOUD DETECTION (Key Metric):")
print("="*70)
print(f"\n{'Variant':<30} {'Thin Cloud Recall':<20} {'vs Baseline':<15}")
print("-"*70)
print(f"{'Baseline CNN':<30} {thin_rec_base*100:>18.2f}% {'-':>14}")
print(f"{'RL: Threshold Only (No Boost)':<30} {thin_rec_no*100:>18.2f}% {(thin_rec_no-thin_rec_base)*100:>+13.2f}%")
print(f"{'RL: Full Model (With Boost)':<30} {thin_rec_full*100:>18.2f}% {(thin_rec_full-thin_rec_base)*100:>+13.2f}%")

print("\n" + "="*70)
print("💡 KEY FINDINGS:")
print("="*70)

boost_contribution = (thin_rec_full - thin_rec_no) * 100
total_improvement = (thin_rec_full - thin_rec_base) * 100

print(f"\n✅ Thin cloud boost contributes: {boost_contribution:+.2f}% points")
print(f"✅ Total RL improvement: {total_improvement:+.2f}% points")

if boost_contribution > total_improvement * 0.5:
    print(f"\n🎯 CONCLUSION: Thin cloud boost is CRITICAL (contributes {boost_contribution/total_improvement*100:.1f}% of improvement)")
elif boost_contribution > 0:
    print(f"\n🎯 CONCLUSION: Thin cloud boost is HELPFUL (contributes {boost_contribution/total_improvement*100:.1f}% of improvement)")
else:
    print(f"\n⚠️  CONCLUSION: Thin cloud boost shows minimal contribution")

print("\n" + "="*70)
print("✅ Ablation study complete!")
print("="*70)
