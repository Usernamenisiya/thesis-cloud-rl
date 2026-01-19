"""
Quick evaluation script for thin cloud detection model.
Loads from checkpoints directory and evaluates on test set.
"""

import glob
import numpy as np
import rasterio
from stable_baselines3 import PPO
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, accuracy_score
from cnn_inference import load_sentinel2_image, get_cloud_mask
from rl_thin_cloud_environment import ThinCloudDetectionEnv

print("🔍 Looking for trained models...")

# Find latest model - check both saved models and checkpoints
model_paths = sorted(glob.glob('models/ppo_thin_cloud_*/model.zip'))
checkpoint_paths = sorted(glob.glob('checkpoints/thin_cloud/thin_cloud_*_steps.zip'))

# Prefer checkpoints (most recent training)
if checkpoint_paths:
    model_path = checkpoint_paths[-1]
    print(f"✅ Loading checkpoint: {model_path}")
    model = PPO.load(model_path)
elif model_paths:
    model_path = model_paths[-1].replace('.zip', '')
    print(f"✅ Loading model: {model_path}")
    model = PPO.load(model_path)
else:
    print("❌ No trained model found. Run training first!")
    exit(1)

# Load test data
data_dir = '/content/drive/MyDrive/Colab_Data/cloudsen12_processed_1000'
image_files = sorted(glob.glob(f'{data_dir}/*_image.tif'))
mask_files = sorted(glob.glob(f'{data_dir}/*_mask.tif'))

# Use 20% for testing (same split as training)
split_idx = int(0.8 * len(image_files))
test_images = image_files[split_idx:]
test_masks = mask_files[split_idx:]

print(f"\n📊 Evaluating on {len(test_images)} test patches...")

# Evaluate
all_gt = []
all_pred_rl = []
all_pred_baseline = []
thin_cloud_gt = []
thin_cloud_pred_rl = []
thin_cloud_pred_baseline = []

for idx in range(min(len(test_images), 100)):  # Limit to 100 for speed
    image = load_sentinel2_image(test_images[idx])
    cnn_prob = get_cloud_mask(image)
    
    with rasterio.open(test_masks[idx]) as src:
        gt = src.read(1)
    
    # Create environment
    env = ThinCloudDetectionEnv(image, cnn_prob, gt, patch_size=64)
    
    # Get predictions
    obs, _ = env.reset()
    predictions = np.zeros_like(gt, dtype=np.uint8)
    baseline = (cnn_prob > 0.5).astype(np.uint8)
    
    for patch_idx in range(min(env.num_patches, 100)):
        action, _ = model.predict(obs, deterministic=True)
        
        i, j = env.current_pos
        ps = env.patch_size
        
        # Apply action
        threshold_delta = np.clip(action[0], -0.2, 0.2)
        thin_boost = np.clip(action[1], 0.0, 0.3)
        
        cnn_patch = cnn_prob[i:i+ps, j:j+ps].copy()
        thin_ind = env.thin_cloud_indicator[i:i+ps, j:j+ps]
        cnn_boosted = np.clip(cnn_patch + thin_ind * thin_boost, 0, 1)
        predictions[i:i+ps, j:j+ps] = (cnn_boosted > 0.5 + threshold_delta).astype(np.uint8)
        
        obs, _, done, _, _ = env.step(action)
        if done:
            break
    
    # Collect metrics
    all_gt.append(gt.flatten())
    all_pred_rl.append(predictions.flatten())
    all_pred_baseline.append(baseline.flatten())
    
    # Thin cloud specific
    thin_cloud_gt.append(env.thin_clouds_gt.flatten())
    thin_cloud_pred_rl.append(predictions[env.thin_clouds_gt > 0])
    thin_cloud_pred_baseline.append(baseline[env.thin_clouds_gt > 0])
    
    if (idx + 1) % 20 == 0:
        print(f"  Processed {idx + 1}/{min(len(test_images), 100)} patches...")

# Compute metrics
gt_all = np.concatenate(all_gt)
pred_rl_all = np.concatenate(all_pred_rl)
pred_baseline_all = np.concatenate(all_pred_baseline)

print("\n" + "="*80)
print("📊 OVERALL CLOUD DETECTION RESULTS")
print("="*80)

print("\n🔵 BASELINE CNN (s2cloudless):")
print(f"  Accuracy:  {accuracy_score(gt_all, pred_baseline_all)*100:.2f}%")
print(f"  Precision: {precision_score(gt_all, pred_baseline_all, zero_division=0)*100:.2f}%")
print(f"  Recall:    {recall_score(gt_all, pred_baseline_all, zero_division=0)*100:.2f}%")
print(f"  F1 Score:  {f1_score(gt_all, pred_baseline_all, zero_division=0)*100:.2f}%")
print(f"  IoU:       {jaccard_score(gt_all, pred_baseline_all, zero_division=0):.4f}")

print("\n🚀 RL REFINED MODEL:")
print(f"  Accuracy:  {accuracy_score(gt_all, pred_rl_all)*100:.2f}%")
print(f"  Precision: {precision_score(gt_all, pred_rl_all, zero_division=0)*100:.2f}%")
print(f"  Recall:    {recall_score(gt_all, pred_rl_all, zero_division=0)*100:.2f}%")
print(f"  F1 Score:  {f1_score(gt_all, pred_rl_all, zero_division=0)*100:.2f}%")
print(f"  IoU:       {jaccard_score(gt_all, pred_rl_all, zero_division=0):.4f}")

# Thin cloud specific
if len(thin_cloud_gt) > 0 and sum([x.sum() for x in thin_cloud_gt]) > 0:
    thin_gt = np.concatenate(thin_cloud_gt)
    if len(thin_cloud_pred_rl) > 0:
        thin_pred_rl = np.concatenate(thin_cloud_pred_rl)
        thin_pred_baseline = np.concatenate(thin_cloud_pred_baseline)
        
        print("\n" + "="*80)
        print("☁️  THIN CLOUD DETECTION (Key Metric)")
        print("="*80)
        
        print("\n🔵 BASELINE CNN:")
        print(f"  IoU:       {jaccard_score(thin_gt, thin_pred_baseline, zero_division=0):.4f}")
        print(f"  Recall:    {recall_score(thin_gt, thin_pred_baseline, zero_division=0)*100:.2f}%")
        
        print("\n🚀 RL REFINED:")
        print(f"  IoU:       {jaccard_score(thin_gt, thin_pred_rl, zero_division=0):.4f}")
        print(f"  Recall:    {recall_score(thin_gt, thin_pred_rl, zero_division=0)*100:.2f}%")
        
        improvement = (jaccard_score(thin_gt, thin_pred_rl, zero_division=0) - 
                      jaccard_score(thin_gt, thin_pred_baseline, zero_division=0))
        print(f"\n✨ IMPROVEMENT: {improvement*100:+.2f}% IoU on thin clouds!")

print("\n" + "="*80)
print("✅ Evaluation complete!")
print("="*80)
