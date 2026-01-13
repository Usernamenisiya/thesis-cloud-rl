"""
Train PPO agent with Multi-Feature RL Environment - THIN CLOUD FOCUS

GOAL: Specifically improve detection of THIN/CIRRUS clouds (CNN's main weakness)

This script trains an RL agent that uses:
- Optical thickness estimation (blue/red ratio, reflectance levels)
- Thin cloud boost action (increases confidence for thin cloud pixels)
- Spectral indices (NDSI, NDVI to filter false positives)
- Modified reward with BIG BONUSES for detecting thin clouds
- Penalties for missing thick clouds or false positives on shadows

Key Innovation: "thin_cloud_boost" action that specifically targets low-reflectance clouds

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
from stable_baselines3.common.callbacks import BaseCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

from cnn_inference import load_sentinel2_image, get_cloud_mask
from rl_multifeature_environment import MultiFeatureRefinementEnv


class ProgressCallback(BaseCallback):
    """Callback to monitor training progress."""
    
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            print(f"Step: {self.n_calls:,}")
        return True


def train_multifeature_rl(
    train_image_files,
    train_mask_files,
    total_timesteps=500000,
    learning_rate=3e-4,
    beta=0.7,
    save_path="models"
):
    """
    Train PPO agent with multi-feature environment.
    
    Args:
        train_image_files: List of training image paths
        train_mask_files: List of training mask paths
        total_timesteps: Total training steps
        learning_rate: PPO learning rate
        beta: F-beta parameter (< 1 emphasizes precision)
        save_path: Directory to save model
    """
    print("\n" + "="*80)
    print("🎯 TRAINING THIN CLOUD DETECTION RL AGENT")
    print("="*80)
    print(f"Training patches: {len(train_image_files)}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"F-beta parameter: {beta} (emphasizes precision)")
    print(f"\nFEATURES:")
    print(f"  - Optical thickness indicators (blue/red ratio, reflectance)")
    print(f"  - Thin cloud classification from ground truth")
    print(f"  - Spectral indices (NDSI, NDVI)")
    print(f"\nACTIONS:")
    print(f"  - threshold_delta: Base threshold adjustment")
    print(f"  - thin_cloud_boost: BOOST for thin cloud pixels (KEY!)")
    print(f"  - spectral_weight: Filter false positives")
    print(f"\nREWARD:")
    print(f"  - BIG BONUS for detecting thin clouds specifically")
    print(f"  - Penalties for false positives on shadows")
    print("="*80 + "\n")
    
    # Create dummy environment for initialization
    print("Loading first patch for environment initialization...")
    image_0 = load_sentinel2_image(train_image_files[0])
    cnn_prob_0 = get_cloud_mask(image_0)
    with rasterio.open(train_mask_files[0]) as src:
        gt_0 = src.read(1)
    
    env = MultiFeatureRefinementEnv(
        image_0, 
        cnn_prob_0, 
        gt_0, 
        patch_size=64,
        baseline_threshold=0.5,
        beta=beta
    )
    
    print(f"✅ Environment created")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.shape}")
    print(f"   Patches per image: {env.num_patches}")
    
    # Initialize PPO
    print("\n🤖 Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs/ppo_multifeature"
    )
    
    print("✅ PPO agent initialized")
    print(f"   Policy: MLP")
    print(f"   Steps per update: 2048")
    print(f"   Batch size: 64")
    
    # Training loop with multiple patches
    print(f"\n🏋️ Starting training on {len(train_image_files)} patches...")
    print("This will take approximately 2-3 hours.\n")
    
    callback = ProgressCallback(check_freq=5000)
    
    steps_per_patch = total_timesteps // len(train_image_files)
    
    for epoch in range(len(train_image_files)):
        img_path = train_image_files[epoch % len(train_image_files)]
        mask_path = train_mask_files[epoch % len(train_image_files)]
        
        # Load patch
        image = load_sentinel2_image(img_path)
        cnn_prob = get_cloud_mask(image)
        with rasterio.open(mask_path) as src:
            ground_truth = src.read(1)
        
        # Create new environment for this patch
        env = MultiFeatureRefinementEnv(
            image, 
            cnn_prob, 
            ground_truth,
            patch_size=64,
            baseline_threshold=0.5,
            beta=beta
        )
        
        model.set_env(env)
        
        print(f"📊 Epoch {epoch+1}/{len(train_image_files)} - Patch: {os.path.basename(img_path)}")
        
        # Train on this patch
        model.learn(
            total_timesteps=steps_per_patch,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=True
        )
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"{save_path}/ppo_multifeature_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    model.save(f"{model_dir}/model")
    
    print(f"\n✅ Model saved to: {model_dir}")
    
    return model, model_dir


def evaluate_multifeature_model(
    model,
    test_image_files,
    test_mask_files,
    beta=0.7,
    results_dir="results"
):
    """
    Evaluate multi-feature RL model on test set.
    
    Args:
        model: Trained PPO model
        test_image_files: List of test image paths
        test_mask_files: List of test mask paths
        beta: F-beta parameter
        results_dir: Directory to save results
    """
    print("\n" + "="*80)
    print("📊 EVALUATING THIN CLOUD DETECTION MODEL")
    print("="*80)
    print(f"Test patches: {len(test_image_files)}")
    print(f"F-beta parameter: {beta}")
    print(f"Will report: Overall + Thin Clouds + Thick Clouds metrics")
    print("="*80 + "\n")
    
    all_gt = []
    all_baseline = []
    all_multifeature = []
    
    # Track thin vs thick cloud performance separately
    all_thin_gt = []
    all_thin_pred = []
    all_thick_gt = []
    all_thick_pred = []
    
    action_stats = {
        'threshold_deltas': [],
        'thin_cloud_boosts': [],  # UPDATED
        'spectral_weights': []
    }
    
    os.makedirs(results_dir, exist_ok=True)
    
    for idx, (img_path, mask_path) in enumerate(zip(test_image_files, test_mask_files)):
        print(f"Processing test patch {idx+1}/{len(test_image_files)}...", end='\r')
        
        # Load data
        image = load_sentinel2_image(img_path)
        cnn_prob = get_cloud_mask(image)
        with rasterio.open(mask_path) as src:
            ground_truth = src.read(1)
        
        # Baseline prediction
        baseline_pred = (cnn_prob > 0.5).astype(np.uint8)
        
        # Multi-feature RL prediction
        env = MultiFeatureRefinementEnv(
            image, cnn_prob, ground_truth,
            patch_size=64, baseline_threshold=0.5, beta=beta
        )
        
        multifeature_pred = np.zeros_like(ground_truth, dtype=np.uint8)
        
        obs, _ = env.reset()
        for _ in range(env.num_patches):
            i, j = env.current_pos
            
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Store action statistics
            action_stats['threshold_deltas'].append(action[0])
            action_stats['thin_cloud_boosts'].append(action[1])  # UPDATED: was texture_weights
            action_stats['spectral_weights'].append(action[2])
            
            # Apply action (same logic as environment)
            threshold_delta = np.clip(action[0], -0.3, 0.3)
            thin_cloud_boost = np.clip(action[1], 0.0, 0.4)  # UPDATED
            spectral_weight = np.clip(action[2], 0.0, 1.0)
            
            # Get predictions (using thin cloud boost logic)
            cnn_patch = cnn_prob[i:i+64, j:j+64].copy()
            adjusted_threshold = np.clip(0.5 + threshold_delta, 0.1, 0.9)
            
            # Apply thin cloud boost
            is_thin_cloud_like = np.logical_and(
                np.logical_and(env.normalized_reflectance[i:i+64, j:j+64] > 1000,
                              env.normalized_reflectance[i:i+64, j:j+64] < 4000),
                env.blue_red_ratio[i:i+64, j:j+64] > 1.05
            )
            cnn_boosted = cnn_patch.copy()
            cnn_boosted[is_thin_cloud_like] += thin_cloud_boost
            cnn_boosted = np.clip(cnn_boosted, 0, 1)
            
            threshold_pred = (cnn_boosted > adjusted_threshold).astype(np.float32)
            
            # Spectral masking
            ndvi_patch = env.ndvi[i:i+64, j:j+64]
            ndsi_patch = env.ndsi[i:i+64, j:j+64]
            spectral_mask = np.logical_or(ndvi_patch < 0.2, ndsi_patch > 0.4).astype(np.float32)
            
            # Combine
            combined = (
                threshold_pred * (1.0 - spectral_weight) +
                threshold_pred * spectral_mask * spectral_weight
            )
            
            multifeature_pred[i:i+64, j:j+64] = (combined > 0.5).astype(np.uint8)
            
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        
        # Collect predictions
        gt_binary = (ground_truth > 0).astype(np.uint8)
        all_gt.append(gt_binary.flatten())
        all_baseline.append(baseline_pred.flatten())
        all_multifeature.append(multifeature_pred.flatten())
        
        # Collect thin vs thick cloud predictions (KEY METRICS!)
        all_thin_gt.append(env.thin_clouds_gt.flatten())
        all_thin_pred.append(multifeature_pred.flatten())  # Will filter later
        all_thick_gt.append(env.thick_clouds_gt.flatten())
        all_thick_pred.append(multifeature_pred.flatten())  # Will filter later
    
    print(f"\nProcessed {len(test_image_files)} test patches")
    
    # Concatenate all predictions
    all_gt = np.concatenate(all_gt)
    all_baseline = np.concatenate(all_baseline)
    all_multifeature = np.concatenate(all_multifeature)
    
    # Concatenate thin/thick cloud arrays
    all_thin_gt = np.concatenate(all_thin_gt)
    all_thin_pred = np.concatenate(all_thin_pred)
    all_thick_gt = np.concatenate(all_thick_gt)
    all_thick_pred = np.concatenate(all_thick_pred)
    
    # Calculate overall metrics
    baseline_metrics = {
        'accuracy': accuracy_score(all_gt, all_baseline),
        'precision': precision_score(all_gt, all_baseline, zero_division=0),
        'recall': recall_score(all_gt, all_baseline, zero_division=0),
        'f1_score': f1_score(all_gt, all_baseline, zero_division=0),
        'fbeta_score': fbeta_score(all_gt, all_baseline, beta=beta, zero_division=0)
    }
    
    multifeature_metrics = {
        'accuracy': accuracy_score(all_gt, all_multifeature),
        'precision': precision_score(all_gt, all_multifeature, zero_division=0),
        'recall': recall_score(all_gt, all_multifeature, zero_division=0),
        'f1_score': f1_score(all_gt, all_multifeature, zero_division=0),
        'fbeta_score': fbeta_score(all_gt, all_multifeature, beta=beta, zero_division=0)
    }
    
    # ============================================================
    # KEY METRICS: THIN CLOUD PERFORMANCE (This is what we care about!)
    # ============================================================
    thin_cloud_metrics = {
        'recall': recall_score(all_thin_gt, np.logical_and(all_thin_pred, all_thin_gt), zero_division=0),
        'precision': precision_score(all_thin_gt, np.logical_and(all_thin_pred, all_thin_gt), zero_division=0),
        'f1_score': f1_score(all_thin_gt, np.logical_and(all_thin_pred, all_thin_gt), zero_division=0),
        'thin_pixels_total': int(all_thin_gt.sum()),
        'thin_pixels_detected': int(np.logical_and(all_thin_pred, all_thin_gt).sum())
    }
    
    thick_cloud_metrics = {
        'recall': recall_score(all_thick_gt, np.logical_and(all_thick_pred, all_thick_gt), zero_division=0),
        'precision': precision_score(all_thick_gt, np.logical_and(all_thick_pred, all_thick_gt), zero_division=0),
        'f1_score': f1_score(all_thick_gt, np.logical_and(all_thick_pred, all_thick_gt), zero_division=0),
        'thick_pixels_total': int(all_thick_gt.sum()),
        'thick_pixels_detected': int(np.logical_and(all_thick_pred, all_thick_gt).sum())
    }
    
    # Action statistics
    action_summary = {
        'threshold_delta': {
            'mean': float(np.mean(action_stats['threshold_deltas'])),
            'std': float(np.std(action_stats['threshold_deltas'])),
            'min': float(np.min(action_stats['threshold_deltas'])),
            'max': float(np.max(action_stats['threshold_deltas']))
        },
        'thin_cloud_boost': {  # UPDATED
            'mean': float(np.mean(action_stats['thin_cloud_boosts'])),
            'std': float(np.std(action_stats['thin_cloud_boosts']))
        },
        'spectral_weight': {
            'mean': float(np.mean(action_stats['spectral_weights'])),
            'std': float(np.std(action_stats['spectral_weights']))
        }
    }
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS - THIN CLOUD DETECTION FOCUS")
    print("="*80)
    
    print(f"\n🧠 Baseline CNN (threshold=0.5):")
    print(f"  Accuracy:  {baseline_metrics['accuracy']:.4f}")
    print(f"  Precision: {baseline_metrics['precision']:.4f}")
    print(f"  Recall:    {baseline_metrics['recall']:.4f}")
    print(f"  F1-Score:  {baseline_metrics['f1_score']:.4f}")
    print(f"  F{beta}-Score: {baseline_metrics['fbeta_score']:.4f}")
    
    print(f"\n🎯 Multi-Feature RL (Overall):")
    print(f"  Accuracy:  {multifeature_metrics['accuracy']:.4f}")
    print(f"  Precision: {multifeature_metrics['precision']:.4f}")
    print(f"  Recall:    {multifeature_metrics['recall']:.4f}")
    print(f"  F1-Score:  {multifeature_metrics['f1_score']:.4f}")
    print(f"  F{beta}-Score: {multifeature_metrics['fbeta_score']:.4f}")
    
    improvement = (multifeature_metrics['f1_score'] - baseline_metrics['f1_score']) / baseline_metrics['f1_score'] * 100
    print(f"\n📈 Overall Improvement: {improvement:+.2f}%")
    
    # KEY METRICS: THIN CLOUDS
    print(f"\n💡 THIN CLOUD PERFORMANCE (KEY!):")
    print(f"  Total Thin Cloud Pixels: {thin_cloud_metrics['thin_pixels_total']:,}")
    print(f"  Thin Clouds Detected: {thin_cloud_metrics['thin_pixels_detected']:,}")
    print(f"  Thin Cloud Recall: {thin_cloud_metrics['recall']:.4f} ({thin_cloud_metrics['recall']*100:.1f}%)")
    print(f"  Thin Cloud Precision: {thin_cloud_metrics['precision']:.4f}")
    print(f"  Thin Cloud F1-Score: {thin_cloud_metrics['f1_score']:.4f}")
    
    print(f"\n☁️ THICK CLOUD PERFORMANCE (Baseline):")
    print(f"  Total Thick Cloud Pixels: {thick_cloud_metrics['thick_pixels_total']:,}")
    print(f"  Thick Clouds Detected: {thick_cloud_metrics['thick_pixels_detected']:,}")
    print(f"  Thick Cloud Recall: {thick_cloud_metrics['recall']:.4f} ({thick_cloud_metrics['recall']*100:.1f}%)")
    
    print(f"\n📊 Action Statistics:")
    print(f"  Threshold Delta: {action_summary['threshold_delta']['mean']:+.4f} ± {action_summary['threshold_delta']['std']:.4f}")
    print(f"                   Range: [{action_summary['threshold_delta']['min']:+.4f}, {action_summary['threshold_delta']['max']:+.4f}]")
    print(f"  Thin Cloud Boost: {action_summary['thin_cloud_boost']['mean']:.4f} ± {action_summary['thin_cloud_boost']['std']:.4f}")
    print(f"  Spectral Weight:  {action_summary['spectral_weight']['mean']:.4f} ± {action_summary['spectral_weight']['std']:.4f}")
    
    # Save results
    results = {
        'baseline_cnn': baseline_metrics,
        'multifeature_rl': multifeature_metrics,
        'thin_cloud_metrics': thin_cloud_metrics,  # KEY ADDITION!
        'thick_cloud_metrics': thick_cloud_metrics,
        'action_statistics': action_summary,
        'improvement_percent': float(improvement),
        'beta_parameter': beta,
        'test_patches': len(test_image_files)
    }
    
    results_file = f"{results_dir}/multifeature_rl_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("="*80 + "\n")
    
    return results


def main():
    """Main training and evaluation pipeline."""
    # Load data
    data_dir = 'data/cloudsen12_processed'
    image_files = sorted(glob.glob(f'{data_dir}/*_image.tif'))
    mask_files = sorted(glob.glob(f'{data_dir}/*_mask.tif'))
    
    # Train/test split (80/20)
    split_idx = int(0.8 * len(image_files))
    train_image_files = image_files[:split_idx]
    train_mask_files = mask_files[:split_idx]
    test_image_files = image_files[split_idx:]
    test_mask_files = mask_files[split_idx:]
    
    print(f"📊 Dataset: {len(image_files)} total patches")
    print(f"   Training: {len(train_image_files)} patches")
    print(f"   Test: {len(test_image_files)} patches")
    
    # Train model
    beta = 0.7  # F-beta parameter (emphasize precision)
    model, model_dir = train_multifeature_rl(
        train_image_files,
        train_mask_files,
        total_timesteps=500000,
        learning_rate=3e-4,
        beta=beta
    )
    
    # Evaluate model
    results = evaluate_multifeature_model(
        model,
        test_image_files,
        test_mask_files,
        beta=beta
    )
    
    print("✅ Training and evaluation complete!")


if __name__ == "__main__":
    main()
