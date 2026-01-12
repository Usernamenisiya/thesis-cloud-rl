"""
Fast PPO Training with Pre-loaded Multiple Patches
Optimized version: loads all patches once to avoid I/O overhead
"""
import numpy as np
import rasterio
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import glob
import os
from pathlib import Path
import time
from rl_environment import CloudMaskRefinementEnv
from s2cloudless import S2PixelCloudDetector
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn

# Custom CNN for 11-channel input (10 bands + 1 CNN probability)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations):
        return self.linear(self.cnn(observations))

class TrainingProgressCallback(BaseCallback):
    def __init__(self, check_freq):
        super().__init__()
        self.check_freq = check_freq
        self.start_time = None
    
    def _on_training_start(self):
        self.start_time = time.time()
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            elapsed = time.time() - self.start_time
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            print(f"Timestep {self.n_calls}: {elapsed:.1f}s elapsed, Mean Reward: {mean_reward:.4f}")
        return True

class FastMultiPatchEnv(gym.Env):
    """
    Optimized: Pre-loads all patches into memory once
    """
    def __init__(self, image_paths, mask_paths, patch_size=64):
        super().__init__()
        self.patch_size = patch_size
        
        print(f"⚡ Pre-loading {len(image_paths)} training patches into RAM...")
        start_time = time.time()
        
        # Pre-load ALL patches and CNN predictions once
        detector = S2PixelCloudDetector(threshold=0.5, all_bands=False, average_over=1, dilation_size=0)
        self.patches = []
        
        for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            if (idx + 1) % 10 == 0:
                print(f"  Loading patch {idx+1}/{len(image_paths)}...")
            
            # Load image
            with rasterio.open(img_path) as src:
                image = src.read()
                image = np.transpose(image, (1, 2, 0))
                image = np.clip(image / 10000.0, 0, 1).astype(np.float32)
            
            # Get CNN prediction once
            image_int = (image * 10000).astype(np.int16)
            image_reshaped = image_int[np.newaxis, ...]
            cnn_prob = detector.get_cloud_probability_maps(image_reshaped)[0].astype(np.float32)
            
            # Load ground truth
            with rasterio.open(mask_path) as src:
                ground_truth = src.read(1)
            
            self.patches.append({
                'image': image,
                'cnn_prob': cnn_prob,
                'ground_truth': ground_truth
            })
        
        elapsed = time.time() - start_time
        print(f"✅ All patches pre-loaded in {elapsed:.1f}s")
        
        # Initialize with first patch
        self.current_patch_idx = 0
        patch_data = self.patches[0]
        self.current_env = CloudMaskRefinementEnv(
            patch_data['image'],
            patch_data['cnn_prob'],
            patch_data['ground_truth'],
            patch_size=patch_size
        )
        
        # Expose required attributes
        self.observation_space = self.current_env.observation_space
        self.action_space = self.current_env.action_space
        
        print(f"🎮 FastMultiPatchEnv ready with {len(self.patches)} pre-loaded patches")
    
    def reset(self, seed=None, options=None):
        """Reset: randomly sample a new patch"""
        if seed is not None:
            np.random.seed(seed)
        
        # 20% chance to switch to a different patch
        if np.random.random() < 0.2:
            new_idx = np.random.randint(0, len(self.patches))
            if new_idx != self.current_patch_idx:
                self.current_patch_idx = new_idx
                patch_data = self.patches[new_idx]
                self.current_env = CloudMaskRefinementEnv(
                    patch_data['image'],
                    patch_data['cnn_prob'],
                    patch_data['ground_truth'],
                    patch_size=self.patch_size
                )
        
        return self.current_env.reset()
    
    def step(self, action):
        return self.current_env.step(action)
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass

def main():
    print("=" * 60)
    print("🤖 Fast PPO Training (Pre-loaded Multi-Patch)")
    print("=" * 60)
    
    # Load CloudSEN12 data
    print("\n📂 Loading CloudSEN12 data...")
    data_dir = "data/cloudsen12_processed"
    image_files = sorted(glob.glob(os.path.join(data_dir, "*_image.tif")))
    mask_files = sorted(glob.glob(os.path.join(data_dir, "*_mask.tif")))
    
    if len(image_files) == 0:
        raise ValueError(f"No processed CloudSEN12 data found in {data_dir}")
    
    print(f"✅ Found {len(image_files)} CloudSEN12 patches")
    
    # 80/20 train/test split
    split_idx = int(0.8 * len(image_files))
    train_image_files = image_files[:split_idx]
    train_mask_files = mask_files[:split_idx]
    test_image_files = image_files[split_idx:]
    test_mask_files = mask_files[split_idx:]
    
    print(f"📊 Train: {len(train_image_files)} patches, Test: {len(test_image_files)} patches")
    
    # Create fast multi-patch environment with pre-loading
    print("\n⚡ Creating Fast Multi-Patch environment...")
    env = FastMultiPatchEnv(train_image_files, train_mask_files, patch_size=64)
    env = DummyVecEnv([lambda: env])
    
    print("\n🎯 Using FAST MULTI-PATCH configuration")
    
    # Create PPO model
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )
    
    print("\n🧠 Creating PPO model with CUSTOM CNN for 11-channel input:")
    print(f"   learning_rate: 0.0003")
    print(f"   n_steps: 4096")
    print(f"   batch_size: 256")
    print(f"   n_epochs: 10")
    print(f"   gamma: 0.99")
    print(f"   gae_lambda: 0.95")
    print(f"   clip_range: 0.2")
    print(f"   clip_range_vf: None")
    print(f"   ent_coef: 0.01")
    print(f"   vf_coef: 0.1")
    print(f"   max_grad_norm: 0.5")
    print(f"   use_sde: False")
    print(f"   sde_sample_freq: -1")
    print(f"   target_kl: None")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.1,
        max_grad_norm=0.5,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=device,
        tensorboard_log="./logs"
    )
    
    # Train
    print("\n" + "=" * 60)
    print("🚀 Starting Fast PPO Training")
    print("=" * 60)
    print(f"Training for 500,000 timesteps on {len(train_image_files)} patches...")
    print("All patches pre-loaded - no I/O overhead during training!")
    
    callback = TrainingProgressCallback(check_freq=10000)
    model.learn(
        total_timesteps=500000,
        callback=callback,
        progress_bar=True
    )
    
    # Save model
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"ppo_multipatch_model_{timestamp}"
    model.save(model_path)
    print(f"\n✅ Model saved: {model_path}")
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("📊 Evaluating on Test Set")
    print("=" * 60)
    
    from cnn_inference import load_sentinel2_image, get_cloud_mask
    
    all_gt = []
    all_cnn = []
    all_ppo = []
    
    print(f"\nEvaluating on {len(test_image_files)} test patches...")
    
    for idx, (img_path, mask_path) in enumerate(zip(test_image_files, test_mask_files)):
        print(f"  Test patch {idx+1}/{len(test_image_files)}", end='\r')
        
        # Load test patch
        test_image = load_sentinel2_image(img_path)
        test_cnn_prob = get_cloud_mask(test_image)
        
        with rasterio.open(mask_path) as src:
            test_gt = src.read(1)
        
        # Create evaluation environment
        eval_env = CloudMaskRefinementEnv(test_image, test_cnn_prob, test_gt, patch_size=64)
        rl_predictions = np.zeros_like(test_gt, dtype=np.uint8)
        
        # Evaluate all patches
        num_patches = len(eval_env.all_positions)
        for patch_idx in range(num_patches):
            obs, _ = eval_env.reset()
            i, j = eval_env.current_pos
            patch_size = eval_env.patch_size
            
            action, _ = model.predict(obs, deterministic=True)
            rl_predictions[i:i+patch_size, j:j+patch_size] = action
            obs, reward, done, truncated, info = eval_env.step(action)
        
        # Collect predictions
        gt_binary = (test_gt > 0).astype(np.uint8)
        cnn_binary = (test_cnn_prob > 0.5).astype(np.uint8)
        rl_binary = (rl_predictions > 0).astype(np.uint8)
        
        all_gt.append(gt_binary.flatten())
        all_cnn.append(cnn_binary.flatten())
        all_ppo.append(rl_binary.flatten())
    
    print(f"\n✅ Evaluation completed on {len(test_image_files)} test patches")
    
    # Combine all test patches
    all_gt = np.concatenate(all_gt)
    all_cnn = np.concatenate(all_cnn)
    all_ppo = np.concatenate(all_ppo)
    
    # Calculate metrics
    cnn_accuracy = accuracy_score(all_gt, all_cnn)
    cnn_precision = precision_score(all_gt, all_cnn, zero_division=0)
    cnn_recall = recall_score(all_gt, all_cnn, zero_division=0)
    cnn_f1 = f1_score(all_gt, all_cnn, zero_division=0)
    
    ppo_accuracy = accuracy_score(all_gt, all_ppo)
    ppo_precision = precision_score(all_gt, all_ppo, zero_division=0)
    ppo_recall = recall_score(all_gt, all_ppo, zero_division=0)
    ppo_f1 = f1_score(all_gt, all_ppo, zero_division=0)
    
    # Calculate improvements
    f1_improvement = ((ppo_f1 - cnn_f1) / cnn_f1 * 100) if cnn_f1 > 0 else 0
    accuracy_improvement = ((ppo_accuracy - cnn_accuracy) / cnn_accuracy * 100) if cnn_accuracy > 0 else 0
    
    print("\n" + "=" * 60)
    print(f"📈 MULTI-PATCH PPO RESULTS")
    print("=" * 60)
    print(f"\n📊 Evaluated on {len(test_image_files)} test patches ({len(all_gt):,} pixels)")
    
    print("\n🧠 CNN Baseline:")
    print(f"  Accuracy:  {cnn_accuracy:.4f} ({cnn_accuracy*100:.2f}%)")
    print(f"  Precision: {cnn_precision:.4f}")
    print(f"  Recall:    {cnn_recall:.4f}")
    print(f"  F1-Score:  {cnn_f1:.4f}")
    
    print("\n🤖 PPO Multi-Patch:")
    print(f"  Accuracy:  {ppo_accuracy:.4f} ({ppo_accuracy*100:.2f}%)")
    print(f"  Precision: {ppo_precision:.4f}")
    print(f"  Recall:    {ppo_recall:.4f}")
    print(f"  F1-Score:  {ppo_f1:.4f}")
    
    print("\n🎯 Improvements:")
    print(f"  F1-Score:  {f1_improvement:+.2f}%")
    print(f"  Accuracy:  {accuracy_improvement:+.2f}%")
    print(f"  Precision: {ppo_precision - cnn_precision:+.4f}")
    print(f"  Recall:    {ppo_recall - cnn_recall:+.4f}")
    print("\n" + "=" * 60)
    
    # Save results
    results = {
        'cnn_baseline': {
            'accuracy': float(cnn_accuracy),
            'precision': float(cnn_precision),
            'recall': float(cnn_recall),
            'f1_score': float(cnn_f1)
        },
        'ppo_multipatch': {
            'accuracy': float(ppo_accuracy),
            'precision': float(ppo_precision),
            'recall': float(ppo_recall),
            'f1_score': float(ppo_f1)
        },
        'improvements': {
            'f1_score_percent': float(f1_improvement),
            'accuracy_percent': float(accuracy_improvement),
            'precision_delta': float(ppo_precision - cnn_precision),
            'recall_delta': float(ppo_recall - cnn_recall)
        },
        'test_info': {
            'num_patches': len(test_image_files),
            'num_pixels': len(all_gt),
            'train_patches': len(train_image_files)
        }
    }
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "ppo_multipatch_results.json"
    
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    print(f"\n🎉 Multi-Patch PPO training complete!")

if __name__ == "__main__":
    main()
