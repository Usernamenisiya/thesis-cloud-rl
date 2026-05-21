"""
Evaluation script for thin cloud detection model.
Properly evaluates s2cloudless baseline vs RL refined model.
"""

import glob
import numpy as np
import rasterio
from stable_baselines3 import PPO
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, accuracy_score
from cnn_inference import load_sentinel2_image, get_cloud_mask
from rl_thin_cloud_environment import ThinCloudDetectionEnv

print("🔍 Looking for trained models...")

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

# Use 20% for testing
split_idx = int(0.8 * len(image_files))
test_images = image_files[split_idx:]
test_masks = mask_files[split_idx:]

print(f"\n📊 Evaluating on {len(test_images)} test patches...")

# Metrics accumulators
baseline_metrics = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
rl_metrics = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
thin_baseline = {'tp': 0, 'fp': 0, 'fn': 0}
thin_rl = {'tp': 0, 'fp': 0, 'fn': 0}

# Use ALL test patches for final evaluation
num_samples = len(test_images)
print(f"   Evaluating on all {num_samples} test patches (this may take a while)...")

# Debug: Track model actions
all_threshold_deltas = []
all_thin_boosts = []

for idx in range(num_samples):
    # Load image and get CNN probability
    image = load_sentinel2_image(test_images[idx])
    cnn_prob = get_cloud_mask(image)
    
    # Debug: Check CNN probability range on first image
    if idx == 0:
        print(f"\n🔍 DEBUG - Image 0:")
        print(f"   Image value range: {image.min():.4f} to {image.max():.4f}")
        print(f"   CNN prob range: {cnn_prob.min():.4f} to {cnn_prob.max():.4f}")
        print(f"   CNN prob mean: {cnn_prob.mean():.4f}")
        print(f"   Pixels > 0.5: {(cnn_prob > 0.5).sum()} / {cnn_prob.size} ({(cnn_prob > 0.5).mean()*100:.1f}%)")
    
    with rasterio.open(test_masks[idx]) as src:
        gt_raw = src.read(1)
    
    # Binarize ground truth
    gt_binary = (gt_raw > 0).astype(np.uint8)
    
    # Baseline prediction (threshold at 0.5)
    baseline_pred = (cnn_prob > 0.5).astype(np.uint8)
    
    # Identify thin clouds: low-moderate CNN probability + ground truth cloud
    # Thin clouds are clouds that CNN is uncertain about (prob 0.2-0.6)
    thin_cloud_mask = (gt_binary == 1) & (cnn_prob >= 0.2) & (cnn_prob <= 0.6)
    
    # RL refined prediction - apply learned thresholds
    # Create environment to get observation
    env = ThinCloudDetectionEnv(image, cnn_prob, gt_binary, patch_size=64)
    
    # Apply model across all patches
    rl_pred = np.zeros_like(gt_binary, dtype=np.uint8)
    
    obs, _ = env.reset()
    for patch_idx in range(env.num_patches):
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        i, j = env.current_pos
        ps = env.patch_size
        
        # Apply action to patch
        threshold_delta = np.clip(action[0], -0.2, 0.2)
        thin_boost = np.clip(action[1], 0.0, 0.3)
        
        # Track actions for debugging
        all_threshold_deltas.append(threshold_delta)
        all_thin_boosts.append(thin_boost)
        
        # Get patch data
        cnn_patch = cnn_prob[i:i+ps, j:j+ps].copy()
        thin_indicator = env.thin_cloud_indicator[i:i+ps, j:j+ps]
        
        # Apply thin cloud boost and adjusted threshold
        boosted_prob = np.clip(cnn_patch + thin_indicator * thin_boost, 0, 1)
        rl_pred[i:i+ps, j:j+ps] = (boosted_prob > (0.5 + threshold_delta)).astype(np.uint8)
        
        # Step to next patch
        obs, _, done, _, _ = env.step(action)
        if done:
            break
    
    # Debug: Check if RL predictions differ from baseline
    if idx == 0:
        diff_pixels = np.sum(rl_pred != baseline_pred)
        print(f"   RL vs Baseline different pixels: {diff_pixels} / {rl_pred.size} ({diff_pixels/rl_pred.size*100:.2f}%)")
        print(f"   Baseline cloud pixels: {baseline_pred.sum()}")
        print(f"   RL cloud pixels: {rl_pred.sum()}")
    
    # Accumulate overall metrics
    baseline_metrics['tp'] += np.sum((baseline_pred == 1) & (gt_binary == 1))
    baseline_metrics['fp'] += np.sum((baseline_pred == 1) & (gt_binary == 0))
    baseline_metrics['tn'] += np.sum((baseline_pred == 0) & (gt_binary == 0))
    baseline_metrics['fn'] += np.sum((baseline_pred == 0) & (gt_binary == 1))
    
    rl_metrics['tp'] += np.sum((rl_pred == 1) & (gt_binary == 1))
    rl_metrics['fp'] += np.sum((rl_pred == 1) & (gt_binary == 0))
    rl_metrics['tn'] += np.sum((rl_pred == 0) & (gt_binary == 0))
    rl_metrics['fn'] += np.sum((rl_pred == 0) & (gt_binary == 1))
    
    # Thin cloud specific metrics (only where thin clouds exist)
    if thin_cloud_mask.sum() > 0:
        thin_baseline['tp'] += np.sum((baseline_pred == 1) & thin_cloud_mask)
        thin_baseline['fn'] += np.sum((baseline_pred == 0) & thin_cloud_mask)
        
        thin_rl['tp'] += np.sum((rl_pred == 1) & thin_cloud_mask)
        thin_rl['fn'] += np.sum((rl_pred == 0) & thin_cloud_mask)
    
    if (idx + 1) % 20 == 0:
        print(f"  Processed {idx + 1}/{num_samples} patches...")

# Debug: Show action statistics
print(f"\n🔍 DEBUG - Model Actions:")
print(f"   Threshold delta: mean={np.mean(all_threshold_deltas):.4f}, range=[{np.min(all_threshold_deltas):.4f}, {np.max(all_threshold_deltas):.4f}]")
print(f"   Thin boost: mean={np.mean(all_thin_boosts):.4f}, range=[{np.min(all_thin_boosts):.4f}, {np.max(all_thin_boosts):.4f}]")

# Compute metrics from confusion matrix
def compute_metrics(m):
    tp, fp, tn, fn = m['tp'], m['fp'], m['tn'], m['fn']
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    return accuracy, precision, recall, f1, iou

print("\n" + "="*80)
print("📊 OVERALL CLOUD DETECTION RESULTS")
print("="*80)

acc, prec, rec, f1, iou = compute_metrics(baseline_metrics)
print("\n🔵 s2cloudless BASELINE:")
print(f"  Accuracy:  {acc*100:.2f}%")
print(f"  Precision: {prec*100:.2f}%")
print(f"  Recall:    {rec*100:.2f}%")
print(f"  F1 Score:  {f1*100:.2f}%")
print(f"  IoU:       {iou:.4f}")

acc_rl, prec_rl, rec_rl, f1_rl, iou_rl = compute_metrics(rl_metrics)
print("\n🚀 RL REFINED MODEL:")
print(f"  Accuracy:  {acc_rl*100:.2f}%")
print(f"  Precision: {prec_rl*100:.2f}%")
print(f"  Recall:    {rec_rl*100:.2f}%")
print(f"  F1 Score:  {f1_rl*100:.2f}%")
print(f"  IoU:       {iou_rl:.4f}")

print("\n📈 IMPROVEMENT:")
print(f"  Accuracy:  {(acc_rl - acc)*100:+.2f}%")
print(f"  Precision: {(prec_rl - prec)*100:+.2f}%")
print(f"  Recall:    {(rec_rl - rec)*100:+.2f}%")
print(f"  F1 Score:  {(f1_rl - f1)*100:+.2f}%")
print(f"  IoU:       {(iou_rl - iou)*100:+.2f}%")

# Thin cloud specific
print("\n" + "="*80)
print("☁️  THIN CLOUD DETECTION (Key Metric)")
print("="*80)

if thin_baseline['tp'] + thin_baseline['fn'] > 0:
    thin_recall_baseline = thin_baseline['tp'] / (thin_baseline['tp'] + thin_baseline['fn'])
    thin_recall_rl = thin_rl['tp'] / (thin_rl['tp'] + thin_rl['fn']) if (thin_rl['tp'] + thin_rl['fn']) > 0 else 0
    
    print(f"\n🔵 s2cloudless Baseline thin cloud recall: {thin_recall_baseline*100:.2f}%")
    print(f"🚀 RL MODEL thin cloud recall:     {thin_recall_rl*100:.2f}%")
    print(f"\n✨ THIN CLOUD IMPROVEMENT: {(thin_recall_rl - thin_recall_baseline)*100:+.2f}%")
else:
    print("\n⚠️  No thin clouds detected in test set")

print("\n" + "="*80)
print("✅ Evaluation complete!")
print("="*80)
