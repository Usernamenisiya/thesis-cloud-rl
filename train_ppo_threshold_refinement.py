"""
Train PPO agent for adaptive threshold refinement
Novel RL approach that learns spatially-varying thresholds
"""

import numpy as np
import glob
import rasterio
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from cnn_inference import load_sentinel2_image, get_cloud_mask
from rl_threshold_environment import ThresholdRefinementEnv


class TrainingCallback(BaseCallback):
    """Callback to log training progress"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.timestep_start = 0
    
    def _on_step(self):
        # Log every 10k timesteps
        if self.num_timesteps % 10000 == 0 and self.num_timesteps != self.timestep_start:
            elapsed = self.num_timesteps - self.timestep_start
            print(f"  Timesteps: {self.num_timesteps:,} | Mean Reward: {np.mean(self.episode_rewards[-100:]):.3f}")
            self.timestep_start = self.num_timesteps
        
        return True
    
    def _on_rollout_end(self):
        # Track episode stats
        if len(self.model.ep_info_buffer) > 0:
            self.episode_rewards.append(self.model.ep_info_buffer[-1]['r'])
            self.episode_lengths.append(self.model.ep_info_buffer[-1]['l'])


def evaluate_threshold_model(model, test_image_files, test_mask_files, patch_size=64, baseline_threshold=0.5):
    """Evaluate trained threshold refinement model on test set"""
    print(f"\n📊 Evaluating threshold model on {len(test_image_files)} test patches...")
    
    all_gt = []
    all_cnn_baseline = []
    all_rl_adjusted = []
    
    threshold_adjustments = []
    
    for idx, (img_path, mask_path) in enumerate(zip(test_image_files, test_mask_files)):
        print(f"  Processing test patch {idx+1}/{len(test_image_files)}", end='\r')
        
        # Load data
        image = load_sentinel2_image(img_path)
        cnn_prob = get_cloud_mask(image)
        
        with rasterio.open(mask_path) as src:
            ground_truth = src.read(1)
        
        # Create environment
        env = ThresholdRefinementEnv(image, cnn_prob, ground_truth, patch_size=patch_size, baseline_threshold=baseline_threshold)
        
        # Initialize prediction arrays
        rl_predictions = np.zeros_like(ground_truth, dtype=np.uint8)
        baseline_predictions = (cnn_prob > baseline_threshold).astype(np.uint8)
        
        # Evaluate all patches
        obs, _ = env.reset()
        
        for patch_idx in range(env.num_patches):
            i, j = env.current_pos
            
            # Get RL action (threshold adjustment)
            action, _ = model.predict(obs, deterministic=True)
            
            # Apply action
            threshold_delta = np.clip(action[0], -0.3, 0.3)
            adjusted_threshold = np.clip(baseline_threshold + threshold_delta, 0.1, 0.9)
            threshold_adjustments.append(threshold_delta)
            
            # Extract patch and apply adjusted threshold
            cnn_patch = cnn_prob[i:i+patch_size, j:j+patch_size]
            rl_patch = (cnn_patch > adjusted_threshold).astype(np.uint8)
            
            # Store predictions
            rl_predictions[i:i+patch_size, j:j+patch_size] = rl_patch
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            if done:
                break
        
        # Collect results
        gt_binary = (ground_truth > 0).astype(np.uint8)
        
        all_gt.append(gt_binary.flatten())
        all_cnn_baseline.append(baseline_predictions.flatten())
        all_rl_adjusted.append(rl_predictions.flatten())
    
    print(f"\n✅ Evaluation completed")
    
    # Combine all patches
    all_gt = np.concatenate(all_gt)
    all_cnn_baseline = np.concatenate(all_cnn_baseline)
    all_rl_adjusted = np.concatenate(all_rl_adjusted)
    
    # Calculate metrics
    cnn_metrics = {
        'accuracy': accuracy_score(all_gt, all_cnn_baseline),
        'precision': precision_score(all_gt, all_cnn_baseline, zero_division=0),
        'recall': recall_score(all_gt, all_cnn_baseline, zero_division=0),
        'f1_score': f1_score(all_gt, all_cnn_baseline, zero_division=0)
    }
    
    rl_metrics = {
        'accuracy': accuracy_score(all_gt, all_rl_adjusted),
        'precision': precision_score(all_gt, all_rl_adjusted, zero_division=0),
        'recall': recall_score(all_gt, all_rl_adjusted, zero_division=0),
        'f1_score': f1_score(all_gt, all_rl_adjusted, zero_division=0)
    }
    
    # Threshold adjustment statistics
    threshold_stats = {
        'mean_delta': float(np.mean(threshold_adjustments)),
        'std_delta': float(np.std(threshold_adjustments)),
        'min_delta': float(np.min(threshold_adjustments)),
        'max_delta': float(np.max(threshold_adjustments))
    }
    
    return cnn_metrics, rl_metrics, threshold_stats


def main():
    print("🎯 RL Adaptive Threshold Refinement Training")
    print("="*60)
    
    # Load data
    data_dir = 'data/cloudsen12_processed'
    image_files = sorted(glob.glob(f'{data_dir}/*_image.tif'))
    mask_files = sorted(glob.glob(f'{data_dir}/*_mask.tif'))
    
    # 80/20 split
    split_idx = int(0.8 * len(image_files))
    train_image_files = image_files[:split_idx]
    train_mask_files = mask_files[:split_idx]
    test_image_files = image_files[split_idx:]
    test_mask_files = mask_files[split_idx:]
    
    print(f"📊 Train patches: {len(train_image_files)}")
    print(f"📊 Test patches: {len(test_image_files)}")
    
    # Load first training patch to create environment
    print("\n🔧 Creating training environment...")
    first_image = load_sentinel2_image(train_image_files[0])
    first_cnn_prob = get_cloud_mask(first_image)
    
    with rasterio.open(train_mask_files[0]) as src:
        first_ground_truth = src.read(1)
    
    env = ThresholdRefinementEnv(
        first_image,
        first_cnn_prob,
        first_ground_truth,
        patch_size=64,
        baseline_threshold=0.5
    )
    
    print(f"✅ Environment created: {env.num_patches} patches per episode")
    
    # Create PPO model
    print("\n🤖 Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device='cuda'
    )
    
    print("✅ PPO model initialized")
    
    # Training callback
    callback = TrainingCallback()
    
    # Train
    total_timesteps = 300000
    print(f"\n🚀 Starting training ({total_timesteps:,} timesteps)...")
    print("="*60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    print("\n✅ Training completed!")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"ppo_threshold_refinement_{timestamp}"
    model.save(model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("📈 EVALUATING ON TEST SET")
    print("="*60)
    
    cnn_metrics, rl_metrics, threshold_stats = evaluate_threshold_model(
        model,
        test_image_files,
        test_mask_files,
        patch_size=64,
        baseline_threshold=0.5
    )
    
    # Calculate improvements
    f1_improvement = ((rl_metrics['f1_score'] - cnn_metrics['f1_score']) / cnn_metrics['f1_score'] * 100) if cnn_metrics['f1_score'] > 0 else 0
    accuracy_improvement = ((rl_metrics['accuracy'] - cnn_metrics['accuracy']) / cnn_metrics['accuracy'] * 100) if cnn_metrics['accuracy'] > 0 else 0
    
    # Display results
    print("\n" + "="*60)
    print(f"📈 THRESHOLD REFINEMENT RESULTS ({len(test_image_files)} test patches)")
    print("="*60)
    
    print("\n🧠 CNN Baseline (threshold=0.5):")
    print(f"  Accuracy:  {cnn_metrics['accuracy']:.4f}")
    print(f"  Precision: {cnn_metrics['precision']:.4f}")
    print(f"  Recall:    {cnn_metrics['recall']:.4f}")
    print(f"  F1-Score:  {cnn_metrics['f1_score']:.4f}")
    
    print("\n🎯 RL Adaptive Threshold:")
    print(f"  Accuracy:  {rl_metrics['accuracy']:.4f}")
    print(f"  Precision: {rl_metrics['precision']:.4f}")
    print(f"  Recall:    {rl_metrics['recall']:.4f}")
    print(f"  F1-Score:  {rl_metrics['f1_score']:.4f}")
    
    print("\n📊 Threshold Adjustment Statistics:")
    print(f"  Mean delta:  {threshold_stats['mean_delta']:+.4f}")
    print(f"  Std delta:   {threshold_stats['std_delta']:.4f}")
    print(f"  Min delta:   {threshold_stats['min_delta']:+.4f}")
    print(f"  Max delta:   {threshold_stats['max_delta']:+.4f}")
    
    print("\n🎯 Improvements:")
    print(f"  F1-Score:  {f1_improvement:+.2f}%")
    print(f"  Accuracy:  {accuracy_improvement:+.2f}%")
    print(f"  Precision: {rl_metrics['precision'] - cnn_metrics['precision']:+.4f}")
    print(f"  Recall:    {rl_metrics['recall'] - cnn_metrics['recall']:+.4f}")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    results = {
        'timestamp': timestamp,
        'cnn_baseline': cnn_metrics,
        'rl_threshold_refinement': rl_metrics,
        'threshold_statistics': threshold_stats,
        'improvements': {
            'f1_score_percent': f1_improvement,
            'accuracy_percent': accuracy_improvement,
            'precision_delta': rl_metrics['precision'] - cnn_metrics['precision'],
            'recall_delta': rl_metrics['recall'] - cnn_metrics['recall']
        },
        'hyperparameters': {
            'total_timesteps': total_timesteps,
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'patch_size': 64,
            'baseline_threshold': 0.5
        }
    }
    
    results_path = results_dir / "threshold_refinement_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")
    print("="*60)

if __name__ == "__main__":
    main()
