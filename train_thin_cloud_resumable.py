"""
Resumable Thin Cloud Detection Training - IMPROVED VERSION

Aligned with best practices:
✅ CNN output as RL state (not raw image)
✅ Patch-level actions (64x64)
✅ Reward = IoU improvement on THIN CLOUDS ONLY
✅ Limited episodes per scene (uses subset per epoch)
✅ PPO with checkpointing

Features:
- Automatic checkpoint detection and resumption
- 100k step sessions (faster iterations)
- Saves every 10k steps to Google Drive
- Uses subset of patches per epoch for efficiency
- Tracks thin cloud IoU improvement specifically

Usage:
    python train_thin_cloud_resumable.py
    
Author: Thesis Implementation
Date: January 2026
"""

import os
import glob
import json
import numpy as np
import rasterio
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score

from cnn_inference import load_sentinel2_image, get_cloud_mask
from rl_thin_cloud_environment import ThinCloudDetectionEnv


class ThinCloudMetricsCallback(BaseCallback):
    """Callback to track thin cloud detection performance."""
    
    def __init__(self, eval_freq=5000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.total_steps = 0  # Track across all learn() calls
        self.rewards = []
        
    def _on_step(self):
        self.total_steps += 1
        
        if self.total_steps % self.eval_freq == 0:
            if len(self.rewards) > 0:
                avg_reward = np.mean(self.rewards[-100:])
                print(f"  Step {self.total_steps:,}: Avg reward = {avg_reward:.4f}", flush=True)
            self.rewards.append(self.locals.get('rewards', [0])[0] if 'rewards' in self.locals else 0)
        
        return True  # Never stop from callback - let the loop handle it


class ProgressTracker:
    """Track training progress across sessions."""
    
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.progress_file = f"{checkpoint_dir}/thin_cloud_progress.json"
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.load_progress()
    
    def load_progress(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
            print(f"📊 Loaded progress: {self.progress['total_steps']:,} steps completed")
        else:
            self.progress = {
                'total_steps': 0,
                'sessions': 0,
                'best_thin_iou': 0.0,
                'history': []
            }
    
    def save_progress(self, steps_this_session, thin_iou=None):
        self.progress['total_steps'] += steps_this_session
        self.progress['sessions'] += 1
        if thin_iou and thin_iou > self.progress['best_thin_iou']:
            self.progress['best_thin_iou'] = thin_iou
        self.progress['history'].append({
            'session': self.progress['sessions'],
            'steps': steps_this_session,
            'timestamp': datetime.now().isoformat(),
            'thin_iou': thin_iou
        })
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)


def find_latest_checkpoint(checkpoint_dir):
    """Find most recent checkpoint."""
    checkpoints = glob.glob(f"{checkpoint_dir}/thin_cloud_*.zip")
    if not checkpoints:
        return None
    # Sort by step number
    def get_step(path):
        try:
            return int(os.path.basename(path).split('_')[-2])
        except:
            return 0
    checkpoints.sort(key=get_step, reverse=True)
    return checkpoints[0]


def evaluate_thin_cloud_performance(model, image_files, mask_files, num_samples=50):
    """
    Evaluate model specifically on thin cloud detection.
    
    Returns thin cloud IoU - THE KEY METRIC.
    """
    print("\n📊 Evaluating thin cloud detection...")
    
    all_thin_gt = []
    all_pred = []
    all_baseline = []
    
    sample_indices = np.random.choice(len(image_files), min(num_samples, len(image_files)), replace=False)
    
    for idx in sample_indices:
        image = load_sentinel2_image(image_files[idx])
        cnn_prob = get_cloud_mask(image)
        
        with rasterio.open(mask_files[idx]) as src:
            gt = src.read(1)
        
        # Create environment (for thin cloud classification)
        env = ThinCloudDetectionEnv(image, cnn_prob, gt, patch_size=64)
        
        # Get predictions
        obs, _ = env.reset()
        predictions = np.zeros_like(gt, dtype=np.uint8)
        baseline = (cnn_prob > 0.5).astype(np.uint8)
        
        for patch_idx in range(min(env.num_patches, 100)):  # Limit patches per scene
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
        
        # Collect thin cloud metrics
        all_thin_gt.append(env.thin_clouds_gt.flatten())
        all_pred.append(predictions.flatten())
        all_baseline.append(baseline.flatten())
    
    # Compute metrics
    thin_gt = np.concatenate(all_thin_gt)
    pred = np.concatenate(all_pred)
    baseline = np.concatenate(all_baseline)
    
    if thin_gt.sum() > 0:
        thin_iou_model = jaccard_score(thin_gt, pred, zero_division=0)
        thin_iou_baseline = jaccard_score(thin_gt, baseline, zero_division=0)
        
        print(f"\n🎯 THIN CLOUD METRICS:")
        print(f"  Baseline IoU:    {thin_iou_baseline:.4f}")
        print(f"  RL Model IoU:    {thin_iou_model:.4f}")
        print(f"  Improvement:     {(thin_iou_model - thin_iou_baseline)*100:+.2f}%")
        
        return thin_iou_model
    else:
        print("⚠️ No thin clouds found in evaluation samples")
        return 0.0


def train_thin_cloud_detection(
    image_files,
    mask_files,
    steps_per_session=100000,
    checkpoint_dir="/content/drive/MyDrive/Colab_Data/thin_cloud_multiobj",  # Save directly to Drive!
    model_dir="models",
    patches_per_epoch=100  # Limit episodes per scene as recommended
):
    """
    Train thin cloud detection with resumability.
    
    Args:
        image_files: Training image paths
        mask_files: Training mask paths  
        steps_per_session: Steps before saving (100k default)
        checkpoint_dir: Where to save checkpoints
        patches_per_epoch: Limit patches per scene (friend's recommendation)
    """
    print("\n" + "="*80)
    print("🎯 THIN CLOUD DETECTION - RESUMABLE TRAINING")
    print("="*80)
    print(f"\n✅ Following best practices:")
    print(f"   - CNN output as state (not raw pixels)")
    print(f"   - Patch-level actions (64x64)")
    print(f"   - Reward = IoU improvement on thin clouds ONLY")
    print(f"   - Limited to {patches_per_epoch} patches per scene")
    print(f"   - PPO algorithm with checkpointing")
    print("="*80)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize tracker
    tracker = ProgressTracker(checkpoint_dir)
    
    # Check for existing checkpoint
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    
    # Create environment with first image
    print("\n📦 Initializing environment...")
    idx = np.random.randint(len(image_files))
    image = load_sentinel2_image(image_files[idx])
    cnn_prob = get_cloud_mask(image)
    with rasterio.open(mask_files[idx]) as src:
        gt = src.read(1)
    
    env = ThinCloudDetectionEnv(image, cnn_prob, gt, patch_size=64)
    vec_env = DummyVecEnv([lambda: env])
    
    # Load or create model
    if latest_checkpoint:
        print(f"📂 Resuming from: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=vec_env)
    else:
        print("🆕 Creating new model...")
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
            tensorboard_log=f"{checkpoint_dir}/tensorboard"
        )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,
        name_prefix="thin_cloud"
    )
    metrics_callback = ThinCloudMetricsCallback(eval_freq=5000)
    
    # Training loop with random scene sampling
    print(f"\n🚀 Training for {steps_per_session:,} steps...")
    print(f"   Using {len(image_files)} training scenes")
    print(f"   {patches_per_epoch} patches per scene max")
    
    total_steps = 0
    scene_count = 0
    
    while total_steps < steps_per_session:
        # Sample random scene
        idx = np.random.randint(len(image_files))
        image = load_sentinel2_image(image_files[idx])
        cnn_prob = get_cloud_mask(image)
        with rasterio.open(mask_files[idx]) as src:
            gt = src.read(1)
        
        # Create fresh environment
        env = ThinCloudDetectionEnv(image, cnn_prob, gt, patch_size=64)
        
        # Limit patches per scene (friend's recommendation: 100-300)
        steps_this_scene = min(patches_per_epoch, env.num_patches)
        
        # Update model's environment
        model.set_env(DummyVecEnv([lambda: env]))
        
        # Train on this scene
        model.learn(
            total_timesteps=steps_this_scene,
            callback=[checkpoint_callback, metrics_callback],
            reset_num_timesteps=False,
            progress_bar=False
        )
        
        total_steps += steps_this_scene
        scene_count += 1
        
        if scene_count % 20 == 0:
            print(f"  Processed {scene_count} scenes, {total_steps:,}/{steps_per_session:,} total steps", flush=True)
    
    # Final save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    final_path = f"{model_dir}/ppo_thin_cloud_{timestamp}"
    model.save(f"{final_path}/model")
    print(f"\n💾 Model saved to: {final_path}")
    
    # Also save to Drive for persistence
    drive_path = "/content/drive/MyDrive/Colab_Data/thin_cloud_final"
    os.makedirs(drive_path, exist_ok=True)
    model.save(f"{drive_path}/model")
    print(f"💾 Model also saved to: {drive_path}")
    
    # Evaluate
    thin_iou = evaluate_thin_cloud_performance(model, image_files, mask_files)
    
    # Save progress
    tracker.save_progress(total_steps, thin_iou)
    
    print(f"\n✅ Session complete!")
    print(f"   Total steps this session: {total_steps:,}")
    print(f"   Total steps overall: {tracker.progress['total_steps']:,}")
    print(f"   Best thin cloud IoU: {tracker.progress['best_thin_iou']:.4f}")
    
    return model, thin_iou


def main():
    """Main entry point."""
    # Detect data location
    data_dirs = [
        '/content/drive/MyDrive/Colab_Data/cloudsen12_processed_1000',  # Colab
        'data/cloudsen12_processed',  # Local
    ]
    
    data_dir = None
    for d in data_dirs:
        if os.path.exists(d):
            data_dir = d
            break
    
    if data_dir is None:
        print("❌ No data directory found!")
        return
    
    image_files = sorted(glob.glob(f'{data_dir}/*_image.tif'))
    mask_files = sorted(glob.glob(f'{data_dir}/*_mask.tif'))
    
    print(f"📂 Found {len(image_files)} images in {data_dir}")
    
    # Use 80% for training
    split_idx = int(0.8 * len(image_files))
    train_images = image_files[:split_idx]
    train_masks = mask_files[:split_idx]
    
    print(f"🎓 Training on {len(train_images)} images")
    
    # Train
    model, thin_iou = train_thin_cloud_detection(
        train_images,
        train_masks,
        steps_per_session=100000,
        patches_per_epoch=100  # As recommended: limit episodes per scene
    )
    
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print(f"   Final thin cloud IoU: {thin_iou:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
