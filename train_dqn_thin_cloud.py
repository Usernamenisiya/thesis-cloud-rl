"""
DQN Training Script for Thin Cloud Detection
Resumable training with checkpointing, comparable to PPO version.

Usage:
    python train_dqn_thin_cloud.py

The script will:
- Auto-resume from latest checkpoint if available
- Save checkpoints every 10,000 steps
- Log training progress
- Save to Google Drive (if in Colab)
"""

import os
import sys
import json
import glob
import numpy as np
from pathlib import Path
from datetime import datetime

# Check if running in Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    CHECKPOINT_DIR = '/content/drive/MyDrive/Colab_Data/dqn_thin_cloud'
    DATA_DIR = '/content/drive/MyDrive/Colab_Data/cloudsen12_processed_1000'
else:
    CHECKPOINT_DIR = 'checkpoints/dqn_thin_cloud'
    DATA_DIR = 'data/cloudsen12_processed_1000'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training parameters
TOTAL_TIMESTEPS = 100_000  # Per session
CHECKPOINT_FREQ = 10_000
TRAIN_SPLIT = 0.8

def load_data():
    """Load CloudSEN12 dataset."""
    from cnn_inference import load_sentinel2_image, get_cloud_mask
    import rasterio
    
    image_files = sorted(glob.glob(f'{DATA_DIR}/*_image.tif'))
    mask_files = sorted(glob.glob(f'{DATA_DIR}/*_mask.tif'))
    
    if len(image_files) == 0:
        raise FileNotFoundError(f"No data found in {DATA_DIR}")
    
    print(f"📂 Found {len(image_files)} image patches")
    
    # Train/test split
    split_idx = int(TRAIN_SPLIT * len(image_files))
    train_images = image_files[:split_idx]
    train_masks = mask_files[:split_idx]
    
    print(f"📊 Training on {len(train_images)} patches")
    
    return train_images, train_masks

def create_env(image_path, mask_path):
    """Create discrete environment for a single image."""
    from cnn_inference import load_sentinel2_image, get_cloud_mask
    from rl_thin_cloud_environment_discrete import ThinCloudDetectionEnvDiscrete
    import rasterio
    
    image = load_sentinel2_image(image_path)
    cnn_prob = get_cloud_mask(image)
    
    with rasterio.open(mask_path) as src:
        gt = src.read(1)
    gt_binary = (gt > 0).astype(np.uint8)
    
    return ThinCloudDetectionEnvDiscrete(image, cnn_prob, gt_binary, patch_size=64)

def find_latest_checkpoint():
    """Find the latest checkpoint to resume from."""
    checkpoints = sorted(glob.glob(f'{CHECKPOINT_DIR}/dqn_thin_cloud_*_steps.zip'))
    if checkpoints:
        latest = checkpoints[-1]
        # Extract step count from filename
        steps = int(latest.split('_')[-2])
        return latest, steps
    return None, 0

def save_progress(total_steps, session_info):
    """Save training progress to JSON."""
    progress_file = f'{CHECKPOINT_DIR}/dqn_progress.json'
    
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            progress = json.load(f)
    else:
        progress = {
            'total_steps': 0,
            'sessions': 0,
            'best_reward': -np.inf,
            'history': []
        }
    
    progress['total_steps'] = total_steps
    progress['sessions'] += 1
    progress['history'].append(session_info)
    
    if session_info.get('mean_reward', -np.inf) > progress['best_reward']:
        progress['best_reward'] = session_info['mean_reward']
    
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    return progress

def train():
    """Main training loop."""
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
    
    print("=" * 70)
    print("🧠 DQN Training for Thin Cloud Detection")
    print("=" * 70)
    
    # Load data
    train_images, train_masks = load_data()
    
    # Find checkpoint to resume from
    checkpoint_path, start_steps = find_latest_checkpoint()
    
    if checkpoint_path:
        print(f"📂 Resuming from: {checkpoint_path}")
        print(f"   Total steps so far: {start_steps:,}")
        model = DQN.load(checkpoint_path)
        # Need to set the environment
        idx = np.random.randint(len(train_images))
        env = create_env(train_images[idx], train_masks[idx])
        model.set_env(env)
    else:
        print("🆕 Starting fresh training")
        # Create initial environment
        idx = np.random.randint(len(train_images))
        env = create_env(train_images[idx], train_masks[idx])
        
        # Create DQN model
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            tau=0.005,  # Soft update coefficient
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            tensorboard_log=f"{CHECKPOINT_DIR}/tensorboard"
        )
    
    # Custom callback to rotate through training images
    class RotateEnvCallback(BaseCallback):
        def __init__(self, train_images, train_masks, rotate_freq=500):
            super().__init__()
            self.train_images = train_images
            self.train_masks = train_masks
            self.rotate_freq = rotate_freq
            self.episode_count = 0
            
        def _on_step(self):
            # Check if episode ended
            if self.locals.get('dones', [False])[0]:
                self.episode_count += 1
                
                if self.episode_count % self.rotate_freq == 0:
                    # Create new environment with different image
                    idx = np.random.randint(len(self.train_images))
                    new_env = create_env(self.train_images[idx], self.train_masks[idx])
                    self.model.set_env(new_env)
                    print(f"   🔄 Rotated to image {idx}")
            return True
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="dqn_thin_cloud",
        save_vecnormalize=True
    )
    
    # Rotate env callback
    rotate_callback = RotateEnvCallback(train_images, train_masks, rotate_freq=10)
    
    # Train
    print(f"\n🚀 Training for {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"   Checkpoints saved to: {CHECKPOINT_DIR}")
    print("=" * 70)
    
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, rotate_callback],
        progress_bar=True,
        reset_num_timesteps=False  # Continue step count
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Save final model
    final_steps = start_steps + TOTAL_TIMESTEPS
    final_path = f"{CHECKPOINT_DIR}/dqn_thin_cloud_{final_steps}_steps"
    model.save(final_path)
    print(f"\n💾 Saved final model: {final_path}.zip")
    
    # Save progress
    session_info = {
        'session': datetime.now().isoformat(),
        'steps': TOTAL_TIMESTEPS,
        'total_steps': final_steps,
        'duration_seconds': duration,
        'mean_reward': float(np.mean(model.ep_info_buffer) if model.ep_info_buffer else 0)
    }
    progress = save_progress(final_steps, session_info)
    
    print("\n" + "=" * 70)
    print("📊 Training Summary")
    print("=" * 70)
    print(f"   Total steps: {final_steps:,}")
    print(f"   Sessions: {progress['sessions']}")
    print(f"   Duration: {duration/60:.1f} minutes")
    print(f"   Checkpoints: {CHECKPOINT_DIR}")
    print("=" * 70)
    
    return model

if __name__ == "__main__":
    train()
