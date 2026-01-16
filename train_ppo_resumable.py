"""
Resumable Training Script for PPO Shadow Detection

Features:
- Automatic checkpoint detection and resumption
- 500k step sessions (safer for connection interruptions)
- Saves every 25k steps to Google Drive
- Tracks cumulative progress across sessions
- Run 4 times to reach 2M total steps

Usage:
    Session 1: python train_ppo_resumable.py
    Session 2: python train_ppo_resumable.py  (auto-resumes)
    Session 3: python train_ppo_resumable.py  (auto-resumes)
    Session 4: python train_ppo_resumable.py  (auto-resumes)

Author: Thesis Implementation - Resumable Training
Date: January 2026
"""

import os
import glob
import numpy as np
import rasterio
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
import json

from cnn_inference import load_sentinel2_image, get_cloud_mask
from rl_shadow_detection_environment import ShadowDetectionEnv


class ProgressTracker:
    """Track training progress across multiple sessions."""
    
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.progress_file = f"{checkpoint_dir}/training_progress.json"
        self.load_progress()
    
    def load_progress(self):
        """Load existing progress or initialize."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
            print(f"📊 Loaded existing progress: {self.progress['total_steps_completed']:,} steps completed")
        else:
            self.progress = {
                'total_steps_completed': 0,
                'total_steps_target': 2_000_000,
                'sessions_completed': 0,
                'current_epoch': 1,
                'action_history': [],
                'session_history': [],
                'started_at': datetime.now().isoformat()
            }
            print("🆕 Starting fresh training")
    
    def save_progress(self):
        """Save current progress."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.progress['last_updated'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def update_session(self, steps_completed, session_num):
        """Update after completing a session."""
        self.progress['total_steps_completed'] += steps_completed
        self.progress['sessions_completed'] = session_num
        self.progress['session_history'].append({
            'session': session_num,
            'steps': steps_completed,
            'timestamp': datetime.now().isoformat()
        })
        self.save_progress()
    
    def is_complete(self):
        """Check if training is complete."""
        return self.progress['total_steps_completed'] >= self.progress['total_steps_target']
    
    def get_remaining_steps(self):
        """Get remaining steps."""
        return self.progress['total_steps_target'] - self.progress['total_steps_completed']


def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent model checkpoint."""
    checkpoints = glob.glob(f"{checkpoint_dir}/rl_model_*_steps.zip")
    if not checkpoints:
        return None
    # Extract step numbers and find the latest
    checkpoints_with_steps = []
    for cp in checkpoints:
        try:
            # Extract number from filename like "rl_model_25000_steps.zip"
            steps = int(cp.split('_')[-2])
            checkpoints_with_steps.append((steps, cp))
        except:
            continue
    
    if checkpoints_with_steps:
        latest = max(checkpoints_with_steps, key=lambda x: x[0])
        return latest[1].replace('.zip', '')  # Return path without .zip
    return None


def train_resumable_session(data_dir=None, 
                            steps_per_session=500_000,
                            checkpoint_freq=25_000,
                            beta=0.7):
    """
    Train one session with automatic checkpoint and resume.
    
    Args:
        data_dir: Directory containing processed CloudSEN12 patches (auto-detects)
        steps_per_session: Steps to train in this session (default 500k)
        checkpoint_freq: Save checkpoint every N steps (default 25k)
        beta: F-beta parameter (< 1 emphasizes precision)
    
    Returns:
        model: Trained PPO model
        progress_tracker: Progress tracking object
    """
    
    # Auto-detect data directory
    if data_dir is None:
        if os.path.exists('/content/drive/MyDrive/Colab_Data/cloudsen12_processed_1000'):
            data_dir = '/content/drive/MyDrive/Colab_Data/cloudsen12_processed_1000'
            print("📁 Google Drive dataset detected (1000 patches)")
        elif os.path.exists('/content/drive/MyDrive/Colab_Data/cloudsen12_processed'):
            data_dir = '/content/drive/MyDrive/Colab_Data/cloudsen12_processed'
            print("📁 Google Drive dataset detected (100 patches)")
        else:
            data_dir = 'data/cloudsen12_processed'
            print("📁 Using local dataset")
    
    # Auto-detect checkpoint directory (prefer Google Drive)
    if os.path.exists('/content/drive/MyDrive'):
        checkpoint_dir = '/content/drive/MyDrive/thesis_checkpoints/shadow_detection_resumable'
        print("💾 Google Drive detected - using persistent storage")
    else:
        checkpoint_dir = 'checkpoints/shadow_detection_resumable'
        print("⚠️  Local storage - checkpoints won't persist if Colab disconnects!")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize progress tracker
    tracker = ProgressTracker(checkpoint_dir)
    
    # Check if training is already complete
    if tracker.is_complete():
        print("\n✅ TRAINING ALREADY COMPLETE!")
        print(f"   Total steps: {tracker.progress['total_steps_completed']:,}")
        print(f"   Sessions: {tracker.progress['sessions_completed']}")
        print("\nTo start fresh, delete the checkpoint directory.")
        return None, tracker
    
    # Calculate this session's details
    current_session = tracker.progress['sessions_completed'] + 1
    remaining_steps = tracker.get_remaining_steps()
    steps_this_session = min(steps_per_session, remaining_steps)
    
    print("\n" + "="*80)
    print("🎯 RESUMABLE TRAINING SESSION")
    print("="*80)
    print(f"Session: {current_session}")
    print(f"Completed: {tracker.progress['total_steps_completed']:,} / {tracker.progress['total_steps_target']:,} steps")
    print(f"Remaining: {remaining_steps:,} steps")
    print(f"This session: {steps_this_session:,} steps")
    print(f"Checkpoint frequency: every {checkpoint_freq:,} steps")
    print(f"Save location: {checkpoint_dir}")
    print("="*80 + "\n")
    
    # Load dataset
    image_files = sorted(glob.glob(f'{data_dir}/*_image.tif'))
    mask_files = sorted(glob.glob(f'{data_dir}/*_mask.tif'))
    
    print(f"📊 Dataset: {len(image_files)} total patches")
    
    # Train/test split (80/20)
    split_idx = int(0.8 * len(image_files))
    train_images = image_files[:split_idx]
    train_masks = mask_files[:split_idx]
    
    print(f"   Training: {len(train_images)} patches")
    print(f"   Test: {len(image_files) - len(train_images)} patches")
    
    # Load a sample image for environment initialization
    print("\nInitializing environment...")
    current_epoch = tracker.progress['current_epoch']
    if current_epoch > len(train_images):
        current_epoch = 1  # Reset if we've cycled through all
    
    sample_idx = (current_epoch - 1) % len(train_images)
    image = load_sentinel2_image(train_images[sample_idx])
    cnn_prob = get_cloud_mask(image)
    with rasterio.open(train_masks[sample_idx]) as src:
        ground_truth = src.read(1)
    
    # Create environment
    env = ShadowDetectionEnv(image, cnn_prob, ground_truth, patch_size=64, beta=beta)
    env = DummyVecEnv([lambda: env])
    
    print(f"✅ Environment created")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.shape}")
    
    # Try to load existing checkpoint
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint:
        print(f"\n🔄 RESUMING from checkpoint: {os.path.basename(latest_checkpoint)}")
        model = PPO.load(latest_checkpoint, env=env, tensorboard_log="./logs/ppo_shadow_resumable/")
        print("✅ Model loaded successfully")
    else:
        print("\n🤖 Initializing new PPO agent...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            verbose=1,
            tensorboard_log="./logs/ppo_shadow_resumable/"
        )
        print("✅ New PPO agent created")
    
    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="rl_model",
        verbose=1
    )
    
    print(f"\n🏋️ Starting training session {current_session}...")
    print(f"   Steps: {steps_this_session:,}")
    print(f"   Checkpoints: every {checkpoint_freq:,} steps")
    print(f"   Estimated time: {steps_this_session / 200000:.1f}-{steps_this_session / 100000:.1f} hours")
    print("\n💡 TIP: If interrupted, just run this script again - it will auto-resume!")
    print("="*80 + "\n")
    
    # Train with dynamic environment updates
    steps_per_epoch = env.get_attr('num_patches')[0] * 2048
    epochs_this_session = max(1, steps_this_session // steps_per_epoch)
    
    steps_trained = 0
    epoch = current_epoch
    
    while steps_trained < steps_this_session and not tracker.is_complete():
        # Cycle through training images
        img_idx = (epoch - 1) % len(train_images)
        img_path = train_images[img_idx]
        mask_path = train_masks[img_idx]
        
        # Load new patch
        image = load_sentinel2_image(img_path)
        cnn_prob = get_cloud_mask(image)
        with rasterio.open(mask_path) as src:
            ground_truth = src.read(1)
        
        # Update environment
        new_env = ShadowDetectionEnv(image, cnn_prob, ground_truth, patch_size=64, beta=beta)
        env.envs[0] = new_env
        model.set_env(env)
        
        steps_remaining = steps_this_session - steps_trained
        steps_to_train = min(steps_per_epoch, steps_remaining)
        
        total_completed = tracker.progress['total_steps_completed'] + steps_trained
        progress_pct = (total_completed / tracker.progress['total_steps_target']) * 100
        
        print(f"📊 Epoch {epoch} | Patch: {os.path.basename(img_path)}")
        print(f"   Session progress: {steps_trained:,}/{steps_this_session:,} | Overall: {progress_pct:.1f}%")
        
        # Train
        model.learn(
            total_timesteps=steps_to_train,
            reset_num_timesteps=False,
            callback=checkpoint_callback
        )
        
        steps_trained += steps_to_train
        epoch += 1
        
        # Update tracker
        tracker.progress['current_epoch'] = epoch
        tracker.save_progress()
    
    # Update session completion
    tracker.update_session(steps_trained, current_session)
    
    # Save final model for this session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = f"{checkpoint_dir}/session_{current_session}_final_{timestamp}"
    model.save(final_model_path)
    
    print("\n" + "="*80)
    print(f"✅ SESSION {current_session} COMPLETE!")
    print("="*80)
    print(f"Steps trained: {steps_trained:,}")
    print(f"Total progress: {tracker.progress['total_steps_completed']:,} / {tracker.progress['total_steps_target']:,}")
    print(f"Completion: {(tracker.progress['total_steps_completed'] / tracker.progress['total_steps_target']) * 100:.1f}%")
    print(f"Final model saved: {final_model_path}")
    
    if tracker.is_complete():
        print("\n🎉 ALL TRAINING COMPLETE! 🎉")
        print(f"   Total steps: {tracker.progress['total_steps_completed']:,}")
        print(f"   Sessions: {tracker.progress['sessions_completed']}")
    else:
        sessions_remaining = np.ceil(remaining_steps / steps_per_session)
        print(f"\n➡️  Run this script {int(sessions_remaining)} more time(s) to complete training")
    
    print("="*80 + "\n")
    
    return model, tracker


def evaluate_final_model(checkpoint_dir, data_dir=None, beta=0.7):
    """
    Evaluate the final trained model.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        data_dir: Directory containing test data
        beta: F-beta parameter
    """
    print("\n" + "="*80)
    print("📊 FINAL MODEL EVALUATION")
    print("="*80)
    
    # Auto-detect data directory
    if data_dir is None:
        if os.path.exists('/content/drive/MyDrive/Colab_Data/cloudsen12_processed_1000'):
            data_dir = '/content/drive/MyDrive/Colab_Data/cloudsen12_processed_1000'
        elif os.path.exists('/content/drive/MyDrive/Colab_Data/cloudsen12_processed'):
            data_dir = '/content/drive/MyDrive/Colab_Data/cloudsen12_processed'
        else:
            data_dir = 'data/cloudsen12_processed'
    
    # Find latest checkpoint
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    if not latest_checkpoint:
        print("❌ No checkpoint found!")
        return None
    
    print(f"Loading model: {os.path.basename(latest_checkpoint)}")
    
    # Load test data
    image_files = sorted(glob.glob(f'{data_dir}/*_image.tif'))
    mask_files = sorted(glob.glob(f'{data_dir}/*_mask.tif'))
    split_idx = int(0.8 * len(image_files))
    test_images = image_files[split_idx:]
    test_masks = mask_files[split_idx:]
    
    print(f"Test set: {len(test_images)} patches")
    
    # Load model
    sample_image = load_sentinel2_image(test_images[0])
    sample_cnn = get_cloud_mask(sample_image)
    with rasterio.open(test_masks[0]) as src:
        sample_gt = src.read(1)
    
    env = ShadowDetectionEnv(sample_image, sample_cnn, sample_gt, patch_size=64, beta=beta)
    model = PPO.load(latest_checkpoint, env=DummyVecEnv([lambda: env]))
    
    # Evaluate
    all_gt = []
    all_cnn = []
    all_rl = []
    
    print("\nEvaluating...")
    for idx, (img_path, mask_path) in enumerate(zip(test_images, test_masks)):
        image = load_sentinel2_image(img_path)
        cnn_prob = get_cloud_mask(image)
        with rasterio.open(mask_path) as src:
            ground_truth = src.read(1)
        
        # CNN baseline
        cnn_binary = (cnn_prob > 0.5).astype(np.uint8)
        
        # RL prediction
        env = ShadowDetectionEnv(image, cnn_prob, ground_truth, patch_size=64, beta=beta)
        rl_pred = np.zeros_like(ground_truth, dtype=np.uint8)
        
        obs, _ = env.reset()
        for _ in range(env.num_patches):
            i, j = env.current_pos
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            
            # Reconstruct prediction
            threshold_delta = np.clip(action[0], -0.3, 0.3)
            shadow_filter_strength = np.clip(action[1], 0.0, 1.0)
            adjusted_threshold = np.clip(0.5 + threshold_delta, 0.1, 0.9)
            
            cnn_patch = cnn_prob[i:i+64, j:j+64]
            initial_pred = (cnn_patch > adjusted_threshold).astype(np.float32)
            
            patch_data = {
                'ndvi': env.ndvi[i:i+64, j:j+64],
                'brightness': env.brightness[i:i+64, j:j+64],
                'edge_strength': env.edge_strength[i:i+64, j:j+64],
                'blue': env.blue_band[i:i+64, j:j+64]
            }
            shadow_score = env._compute_shadow_likelihood(patch_data)
            shadow_mask = (shadow_score < (1.0 - shadow_filter_strength)).astype(np.float32)
            
            rl_pred[i:i+64, j:j+64] = (initial_pred * shadow_mask).astype(np.uint8)
            
            if done:
                break
        
        gt_binary = (ground_truth > 0).astype(np.uint8)
        all_gt.append(gt_binary.flatten())
        all_cnn.append(cnn_binary.flatten())
        all_rl.append(rl_pred.flatten())
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(test_images)} patches")
    
    # Calculate metrics
    all_gt = np.concatenate(all_gt)
    all_cnn = np.concatenate(all_cnn)
    all_rl = np.concatenate(all_rl)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    # Baseline CNN
    cnn_metrics = {
        'accuracy': accuracy_score(all_gt, all_cnn),
        'precision': precision_score(all_gt, all_cnn, zero_division=0),
        'recall': recall_score(all_gt, all_cnn, zero_division=0),
        'f1_score': f1_score(all_gt, all_cnn, zero_division=0),
        'fbeta_score': fbeta_score(all_gt, all_cnn, beta=beta, zero_division=0)
    }
    
    print(f"\n🧠 Baseline CNN:")
    print(f"  Precision: {cnn_metrics['precision']:.4f}")
    print(f"  Recall:    {cnn_metrics['recall']:.4f}")
    print(f"  F1-Score:  {cnn_metrics['f1_score']:.4f}")
    
    # RL model
    rl_metrics = {
        'accuracy': accuracy_score(all_gt, all_rl),
        'precision': precision_score(all_gt, all_rl, zero_division=0),
        'recall': recall_score(all_gt, all_rl, zero_division=0),
        'f1_score': f1_score(all_gt, all_rl, zero_division=0),
        'fbeta_score': fbeta_score(all_gt, all_rl, beta=beta, zero_division=0)
    }
    
    print(f"\n🎯 Shadow Detection RL:")
    print(f"  Precision: {rl_metrics['precision']:.4f} ({rl_metrics['precision'] - cnn_metrics['precision']:+.4f})")
    print(f"  Recall:    {rl_metrics['recall']:.4f} ({rl_metrics['recall'] - cnn_metrics['recall']:+.4f})")
    print(f"  F1-Score:  {rl_metrics['f1_score']:.4f} ({rl_metrics['f1_score'] - cnn_metrics['f1_score']:+.4f})")
    
    f1_improvement = ((rl_metrics['f1_score'] - cnn_metrics['f1_score']) / cnn_metrics['f1_score'] * 100)
    print(f"\n📈 F1 Improvement: {f1_improvement:+.2f}%")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results = {
        'baseline_cnn': cnn_metrics,
        'shadow_detection_rl': rl_metrics,
        'improvement_percent': f1_improvement,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('results/resumable_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n💾 Results saved to: results/resumable_training_results.json")
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Check for evaluation mode
    if len(sys.argv) > 1 and sys.argv[1] == '--evaluate':
        if os.path.exists('/content/drive/MyDrive'):
            checkpoint_dir = '/content/drive/MyDrive/thesis_checkpoints/shadow_detection_resumable'
        else:
            checkpoint_dir = 'checkpoints/shadow_detection_resumable'
        
        evaluate_final_model(checkpoint_dir)
    else:
        # Normal training mode
        model, tracker = train_resumable_session()
        
        # If training is complete, offer to evaluate
        if tracker and tracker.is_complete():
            print("\n💡 Training complete! Run with --evaluate flag to test the model:")
            print("   python train_ppo_resumable.py --evaluate")
