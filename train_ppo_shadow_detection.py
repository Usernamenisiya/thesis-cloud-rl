"""
Phase 2: Train PPO Agent for Shadow Detection and Filtering

Goal: Improve precision by learning to filter shadow false positives
Starting baseline: RL Threshold (30.25% F1, 20% precision)
Target: 35-40% F1 with >30% precision

Strategy:
- Use shadow indicators (NDVI, brightness, edges) to identify shadows
- Learn optimal shadow filtering strength
- Balance precision improvement with recall preservation

Author: Thesis Implementation - Phase 2
Date: January 2026
"""

import os
import glob
import numpy as np
import rasterio
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
import json

from cnn_inference import load_sentinel2_image, get_cloud_mask
from rl_shadow_detection_environment import ShadowDetectionEnv


def train_shadow_detection_rl(data_dir='data/cloudsen12_processed', 
                              total_timesteps=500000,
                              beta=0.7,
                              checkpoint_dir='checkpoints/shadow_detection',
                              resume_from=None,
                              checkpoint_freq=10):
    """
    Train PPO agent with shadow detection and filtering.
    
    Args:
        data_dir: Directory containing processed CloudSEN12 patches
        total_timesteps: Number of training steps
        beta: F-beta parameter (< 1 emphasizes precision)
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from (e.g., 'checkpoints/shadow_detection/checkpoint_epoch_20')
        checkpoint_freq: Save checkpoint every N epochs
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Load all patches
    image_files = sorted(glob.glob(f'{data_dir}/*_image.tif'))
    mask_files = sorted(glob.glob(f'{data_dir}/*_mask.tif'))
    
    print(f"📊 Dataset: {len(image_files)} total patches")
    
    # Train/test split (80/20)
    split_idx = int(0.8 * len(image_files))
    train_images = image_files[:split_idx]
    train_masks = mask_files[:split_idx]
    test_images = image_files[split_idx:]
    test_masks = mask_files[split_idx:]
    
    # Check for resume
    start_epoch = 1
    model = None
    action_history = []
    
    if resume_from:
        print(f"\n🔄 RESUMING from checkpoint: {resume_from}")
        checkpoint_state = f"{resume_from}_state.json"
        if os.path.exists(checkpoint_state):
            with open(checkpoint_state, 'r') as f:
                state = json.load(f)
                start_epoch = state['epoch'] + 1
                action_history = state.get('action_history', [])
            print(f"   Resuming from epoch {start_epoch}/{len(train_images)}")
        else:
            print(f"   ⚠️ State file not found, starting from epoch {start_epoch}")
    
    print(f"   Training: {len(train_images)} patches")
    print(f"   Test: {len(test_images)} patches")
    
    print("\n" + "="*80)
    print("🎯 TRAINING SHADOW DETECTION RL AGENT (FIXED V2)")
    print("="*80)
    print(f"Training patches: {len(train_images)}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Learning rate: 0.0003")
    print(f"F-beta parameter: {beta} (emphasizes precision)")
    print("\nFEATURES:")
    print("  - NDVI (vegetation visible in shadows, not clouds)")
    print("  - Brightness (shadows very dark)")
    print("  - Edge strength (shadows have sharp edges)")
    print("  - Blue scattering (clouds scatter blue, shadows don't)")
    print("\nACTIONS (CONSTRAINED):")
    print("  - threshold_delta: [-0.15, +0.15] (was -0.3/+0.3)")
    print("  - shadow_filter_strength: [0, 0.6] (was 0/1.0)")
    print("\nREWARD (REDESIGNED):")
    print("  - Base: F-beta * 10")
    print("  - CATASTROPHIC penalties for predict-nothing/everything (-50)")
    print("  - Hard floors: recall >0.15, precision >0.10")
    print("  - Bonuses for balanced improvement (precision + recall)")
    print("  - Action regularization (penalize extremes)")
    print("="*80)
    
    # Load first training patch for environment initialization
    print("\nLoading first patch for environment initialization...")
    image = load_sentinel2_image(train_images[0])
    cnn_prob = get_cloud_mask(image)
    with rasterio.open(train_masks[0]) as src:
        ground_truth = src.read(1)
    
    # Create environment
    env = ShadowDetectionEnv(image, cnn_prob, ground_truth, patch_size=64, beta=beta)
    env = DummyVecEnv([lambda: env])
    
    print(f"✅ Environment created")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.shape}")
    print(f"   Patches per image: {env.get_attr('num_patches')[0]}")
    
    # Create or load PPO agent
    if resume_from and os.path.exists(f"{resume_from}.zip"):
        print(f"\n🤖 Loading model from checkpoint...")
        model = PPO.load(resume_from, env=env, tensorboard_log="./logs/ppo_shadow_detection/")
        print("✅ Model loaded from checkpoint")
    else:
        print("\n🤖 Initializing new PPO agent...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            verbose=1,
            tensorboard_log="./logs/ppo_shadow_detection/"
        )
        print("✅ PPO agent initialized")
    print(f"   Policy: MLP")
    print(f"   Steps per update: 2048")
    print(f"   Batch size: 64")
    
    # Training loop
    print(f"\n🏋️ Starting training on {len(train_images)} patches...")
    if start_epoch > 1:
        print(f"   Resuming from epoch {start_epoch}")
    print("This will take approximately 2-3 hours.")
    print(f"\n💾 CHECKPOINTS: Saving every {checkpoint_freq} epochs to {checkpoint_dir}")
    print("⚠️  MONITORING: Will check for degenerate policies every 10 epochs")
    print()
    
    steps_per_patch = env.get_attr('num_patches')[0] * 2048
    
    # Training data (skip already trained epochs if resuming)
    training_pairs = list(enumerate(zip(train_images, train_masks), 1))
    if start_epoch > 1:
        training_pairs = [(e, (i, m)) for e, (i, m) in training_pairs if e >= start_epoch]
    
    for epoch, (img_path, mask_path) in training_pairs:
        # Load new patch
        image = load_sentinel2_image(img_path)
        cnn_prob = get_cloud_mask(image)
        with rasterio.open(mask_path) as src:
            ground_truth = src.read(1)
        
        # Update environment
        new_env = ShadowDetectionEnv(image, cnn_prob, ground_truth, patch_size=64, beta=beta)
        env.envs[0] = new_env
        model.set_env(env)
        
        print(f"📊 Epoch {epoch}/{len(train_images)} - Patch: {os.path.basename(img_path)}")
        
        # Train on this patch
        model.learn(total_timesteps=steps_per_patch, reset_num_timesteps=False)
        
        # SAFETY CHECK: Monitor actions every 10 epochs
        if epoch % 10 == 0:
            obs = env.reset()
            actions_sample = []
            for _ in range(50):  # Sample 50 actions
                action, _ = model.predict(obs, deterministic=False)
                actions_sample.append(action[0])
                obs, _, done, _, _ = env.step(action)
                if done[0]:
                    obs = env.reset()
            
            actions_array = np.array(actions_sample)
            threshold_mean = actions_array[:, 0].mean()
            threshold_std = actions_array[:, 0].std()
            shadow_mean = actions_array[:, 1].mean()
            shadow_std = actions_array[:, 1].std()
            
            action_history.append({
                'epoch': epoch,
                'threshold_delta': {'mean': float(threshold_mean), 'std': float(threshold_std)},
                'shadow_filter': {'mean': float(shadow_mean), 'std': float(shadow_std)}
            })
            
            print(f"  🔍 Action Check @ Epoch {epoch}:")
            print(f"     Threshold Delta: {threshold_mean:.3f} ± {threshold_std:.3f} (range: [{actions_array[:, 0].min():.3f}, {actions_array[:, 0].max():.3f}])")
            print(f"     Shadow Filter: {shadow_mean:.3f} ± {shadow_std:.3f} (range: [{actions_array[:, 1].min():.3f}, {actions_array[:, 1].max():.3f}])")
            
            # WARNING if converging to extremes
            if threshold_std < 0.02 and (abs(threshold_mean + 0.15) < 0.01 or abs(threshold_mean - 0.15) < 0.01):
                print(f"     ⚠️  WARNING: Threshold delta converging to extreme ({threshold_mean:.3f})")
            if shadow_std < 0.02 and (shadow_mean < 0.01 or shadow_mean > 0.59):
                print(f"     ⚠️  WARNING: Shadow filter converging to extreme ({shadow_mean:.3f})")
            if threshold_std < 0.02 and shadow_std < 0.02:
                print(f"     ⚠️  WARNING: Both actions have very low variance - possible degenerate policy!")
        
        # CHECKPOINT: Save every N epochs
        if epoch % checkpoint_freq == 0:
            checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}"
            model.save(checkpoint_path)
            
            # Save training state
            state = {
                'epoch': epoch,
                'total_epochs': len(train_images),
                'action_history': action_history,
                'timestamp': datetime.now().isoformat()
            }
            with open(f"{checkpoint_path}_state.json", 'w') as f:
                json.dump(state, f, indent=2)
            
            print(f"  💾 Checkpoint saved: {checkpoint_path}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/ppo_shadow_detection_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    model.save(f"{model_dir}/model")
    
    # Save action history
    with open(f"{model_dir}/action_history.json", 'w') as f:
        json.dump(action_history, f, indent=2)
    
    print(f"\n💾 Model saved to: {model_dir}")
    print(f"💾 Action history saved for analysis")
    
    # Evaluation
    print("\n" + "="*80)
    print("📊 EVALUATING ON TEST SET")
    print("="*80)
    print(f"Test patches: {len(test_images)}")
    print(f"F-beta parameter: {beta}")
    print("Will report: Overall + Precision Focus metrics")
    print("="*80 + "\n")
    
    results = evaluate_shadow_detection_model(model, test_images, test_masks, beta=beta)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/shadow_detection_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, results


def evaluate_shadow_detection_model(model, test_images, test_masks, beta=0.7):
    """Evaluate shadow detection model on test set."""
    
    # Baseline CNN metrics
    all_gt = []
    all_cnn = []
    
    # Shadow detection RL metrics
    all_rl = []
    
    # Action statistics
    threshold_deltas = []
    shadow_strengths = []
    
    for img_path, mask_path in zip(test_images, test_masks):
        # Load data
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
            
            # Track actions
            threshold_deltas.append(action[0])
            shadow_strengths.append(action[1])
            
            obs, _, done, _, _ = env.step(action)
            
            # Reconstruct prediction for this patch
            threshold_delta = np.clip(action[0], -0.3, 0.3)
            shadow_filter_strength = np.clip(action[1], 0.0, 1.0)
            adjusted_threshold = np.clip(0.5 + threshold_delta, 0.1, 0.9)
            
            cnn_patch = cnn_prob[i:i+64, j:j+64]
            initial_pred = (cnn_patch > adjusted_threshold).astype(np.float32)
            
            # Apply shadow filtering
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
        
        # Collect predictions
        gt_binary = (ground_truth > 0).astype(np.uint8)
        all_gt.append(gt_binary.flatten())
        all_cnn.append(cnn_binary.flatten())
        all_rl.append(rl_pred.flatten())
    
    # Concatenate all
    all_gt = np.concatenate(all_gt)
    all_cnn = np.concatenate(all_cnn)
    all_rl = np.concatenate(all_rl)
    
    print(f"Processed {len(test_images)} test patches\n")
    
    # Calculate metrics
    print("="*80)
    print("RESULTS - SHADOW DETECTION FOCUS")
    print("="*80)
    
    # Baseline CNN
    cnn_metrics = {
        'accuracy': accuracy_score(all_gt, all_cnn),
        'precision': precision_score(all_gt, all_cnn, zero_division=0),
        'recall': recall_score(all_gt, all_cnn, zero_division=0),
        'f1_score': f1_score(all_gt, all_cnn, zero_division=0),
        'fbeta_score': fbeta_score(all_gt, all_cnn, beta=beta, zero_division=0)
    }
    
    print(f"\n🧠 Baseline CNN (threshold=0.5):")
    print(f"  Accuracy:  {cnn_metrics['accuracy']:.4f}")
    print(f"  Precision: {cnn_metrics['precision']:.4f}")
    print(f"  Recall:    {cnn_metrics['recall']:.4f}")
    print(f"  F1-Score:  {cnn_metrics['f1_score']:.4f}")
    print(f"  F{beta}-Score: {cnn_metrics['fbeta_score']:.4f}")
    
    # Shadow detection RL
    rl_metrics = {
        'accuracy': accuracy_score(all_gt, all_rl),
        'precision': precision_score(all_gt, all_rl, zero_division=0),
        'recall': recall_score(all_gt, all_rl, zero_division=0),
        'f1_score': f1_score(all_gt, all_rl, zero_division=0),
        'fbeta_score': fbeta_score(all_gt, all_rl, beta=beta, zero_division=0)
    }
    
    print(f"\n🎯 Shadow Detection RL:")
    print(f"  Accuracy:  {rl_metrics['accuracy']:.4f}")
    print(f"  Precision: {rl_metrics['precision']:.4f} {'✅' if rl_metrics['precision'] > cnn_metrics['precision'] else '❌'}")
    print(f"  Recall:    {rl_metrics['recall']:.4f}")
    print(f"  F1-Score:  {rl_metrics['f1_score']:.4f}")
    print(f"  F{beta}-Score: {rl_metrics['fbeta_score']:.4f}")
    
    # Improvement
    f1_improvement = ((rl_metrics['f1_score'] - cnn_metrics['f1_score']) / cnn_metrics['f1_score'] * 100)
    precision_improvement = ((rl_metrics['precision'] - cnn_metrics['precision']) / cnn_metrics['precision'] * 100)
    
    print(f"\n📈 Improvement:")
    print(f"  F1-Score: {f1_improvement:+.2f}%")
    print(f"  Precision: {precision_improvement:+.2f}% (KEY METRIC!)")
    
    # Action statistics
    threshold_deltas = np.array(threshold_deltas)
    shadow_strengths = np.array(shadow_strengths)
    
    print(f"\n📊 Action Statistics:")
    print(f"  Threshold Delta: {threshold_deltas.mean():.4f} ± {threshold_deltas.std():.4f}")
    print(f"                   Range: [{threshold_deltas.min():.4f}, {threshold_deltas.max():.4f}]")
    print(f"  Shadow Filter:   {shadow_strengths.mean():.4f} ± {shadow_strengths.std():.4f}")
    print(f"                   Range: [{shadow_strengths.min():.4f}, {shadow_strengths.max():.4f}]")
    
    # Results dict
    results = {
        'baseline_cnn': cnn_metrics,
        'shadow_detection_rl': rl_metrics,
        'improvement_percent': f1_improvement,
        'precision_improvement_percent': precision_improvement,
        'action_statistics': {
            'threshold_delta': {
                'mean': float(threshold_deltas.mean()),
                'std': float(threshold_deltas.std()),
                'min': float(threshold_deltas.min()),
                'max': float(threshold_deltas.max())
            },
            'shadow_filter_strength': {
                'mean': float(shadow_strengths.mean()),
                'std': float(shadow_strengths.std()),
                'min': float(shadow_strengths.min()),
                'max': float(shadow_strengths.max())
            }
        }
    }
    
    print("\n💾 Results saved to: results/shadow_detection_results.json")
    print("="*80)
    
    print("\n✅ Training and evaluation complete!")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Check for resume argument
    resume_from = None
    if len(sys.argv) > 1 and sys.argv[1] == '--resume':
        if len(sys.argv) > 2:
            resume_from = sys.argv[2]
            print(f"\n🔄 Resume mode: {resume_from}")
        else:
            print("\n⚠️  Usage: python train_ppo_shadow_detection.py --resume <checkpoint_path>")
            print("   Example: python train_ppo_shadow_detection.py --resume checkpoints/shadow_detection/checkpoint_epoch_20")
            sys.exit(1)
    
    model, results = train_shadow_detection_rl(resume_from=resume_from)
        data_dir='data/cloudsen12_processed',
        total_timesteps=500000,
        beta=0.7
    )
