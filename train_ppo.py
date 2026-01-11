import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import time
import json
from pathlib import Path
from cnn_inference import load_sentinel2_image, get_cloud_mask
from rl_environment import CloudMaskRefinementEnv
import rasterio
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gymnasium as gym


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for multi-channel satellite imagery.
    Handles 11-channel input (10 Sentinel-2 bands + 1 CNN probability).
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]  # 11 channels
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class TrainingProgressCallback(BaseCallback):
    """Custom callback to monitor training progress"""
    def __init__(self, check_freq=5000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            elapsed = time.time() - self.start_time
            mean_reward = self.model.last_mean_reward if hasattr(self.model, 'last_mean_reward') else 0
            print(f"Timestep {self.num_timesteps}: {elapsed:.1f}s elapsed, Mean Reward: {mean_reward:.4f}")
        return True

def train_ppo():
    """Train PPO agent for cloud mask refinement"""
    
    print("=" * 60)
    print("🤖 Training PPO Agent for Cloud Mask Refinement")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading data files...")
    image = load_sentinel2_image('data/sentinel2_image.tif')
    cnn_prob = get_cloud_mask(image)
    
    with rasterio.open('data/ground_truth.tif') as src:
        ground_truth = src.read(1)
    
    print(f"✅ Data loaded: Image shape {image.shape}, CNN prob shape {cnn_prob.shape}")
    
    # Create environment
    print("\n🎮 Creating RL environment...")
    env = CloudMaskRefinementEnv(image, cnn_prob, ground_truth, patch_size=64)
    
    # PPO configuration - AGGRESSIVE for forcing cloud detection
    ppo_config = {
        "learning_rate": 5e-4,          # Higher LR for faster adaptation
        "n_steps": 1024,                # Smaller rollout for faster updates
        "batch_size": 64,               # Mini-batch size
        "n_epochs": 10,                 # Number of epochs for SGD
        "gamma": 0.99,                  # Discount factor
        "gae_lambda": 0.95,             # GAE lambda for advantage estimation
        "clip_range": 0.3,              # More aggressive clipping
        "clip_range_vf": None,          # Clip value function
        "ent_coef": 0.05,               # HIGHER entropy - more exploration
        "vf_coef": 0.5,                 # Value function coefficient
        "max_grad_norm": 0.5,           # Gradient clipping
        "use_sde": False,               # State-dependent exploration
        "sde_sample_freq": -1,
        "target_kl": None,              # Target KL divergence
        "policy_kwargs": {
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),  # Updated format for SB3 v1.8+
            "activation_fn": torch.nn.ReLU,
            "normalize_images": False,  # We handle normalization ourselves
        }
    }
    
    # Create PPO model
    print("\n🧠 Creating PPO model with CUSTOM CNN for 11-channel input:")
    for key, value in ppo_config.items():
        if key != "policy_kwargs":
            print(f"   {key}: {value}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    
    model = PPO(
        'CnnPolicy',
        env,
        learning_rate=ppo_config["learning_rate"],
        n_steps=ppo_config["n_steps"],
        batch_size=ppo_config["batch_size"],
        n_epochs=ppo_config["n_epochs"],
        gamma=ppo_config["gamma"],
        gae_lambda=ppo_config["gae_lambda"],
        clip_range=ppo_config["clip_range"],
        clip_range_vf=ppo_config["clip_range_vf"],
        ent_coef=ppo_config["ent_coef"],
        vf_coef=ppo_config["vf_coef"],
        max_grad_norm=ppo_config["max_grad_norm"],
        use_sde=ppo_config["use_sde"],
        policy_kwargs=ppo_config["policy_kwargs"],
        verbose=1,
        device=device,
        tensorboard_log="./logs/"
    )
    
    # Train the model
    print("\n" + "=" * 60)
    print("🚀 Starting PPO Training")
    print("=" * 60)
    
    total_timesteps = 100000
    print(f"Training for {total_timesteps:,} timesteps...")
    print("This will take approximately 1-2 hours with GPU...")
    
    callback = TrainingProgressCallback(check_freq=5000)
    
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    training_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print(f"✅ Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    print("=" * 60)
    
    # Save the model
    model_path = "models/ppo_cloud_refinement_model"
    Path("models").mkdir(exist_ok=True)
    model.save(model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Evaluate on test data
    print("\n" + "=" * 60)
    print("📊 Evaluating PPO Agent")
    print("=" * 60)
    
    eval_env = CloudMaskRefinementEnv(image, cnn_prob, ground_truth, patch_size=64)
    rl_predictions = np.zeros_like(ground_truth, dtype=np.uint8)
    
    obs, _ = eval_env.reset()  # Gymnasium returns (obs, info)
    done = False
    step_count = 0
    
    print("\nGenerating predictions...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)  # Gymnasium returns 5 values
        
        if 'patch_position' in info:
            row, col = info['patch_position']
            patch_size = eval_env.patch_size
            rl_predictions[row:row+patch_size, col:col+patch_size] = action
        
        step_count += 1
        if step_count % 10000 == 0:
            print(f"  Evaluation step: {step_count}")
    
    print(f"✅ Evaluation completed in {step_count} steps")
    
    # Calculate metrics
    gt_binary = (ground_truth > 0).astype(np.uint8)
    cnn_binary = (cnn_prob > 0.5).astype(np.uint8)
    rl_binary = (rl_predictions > 0).astype(np.uint8)
    
    # CNN metrics
    cnn_accuracy = accuracy_score(gt_binary.flatten(), cnn_binary.flatten())
    cnn_precision = precision_score(gt_binary.flatten(), cnn_binary.flatten(), zero_division=0)
    cnn_recall = recall_score(gt_binary.flatten(), cnn_binary.flatten(), zero_division=0)
    cnn_f1 = f1_score(gt_binary.flatten(), cnn_binary.flatten(), zero_division=0)
    
    # PPO metrics
    ppo_accuracy = accuracy_score(gt_binary.flatten(), rl_binary.flatten())
    ppo_precision = precision_score(gt_binary.flatten(), rl_binary.flatten(), zero_division=0)
    ppo_recall = recall_score(gt_binary.flatten(), rl_binary.flatten(), zero_division=0)
    ppo_f1 = f1_score(gt_binary.flatten(), rl_binary.flatten(), zero_division=0)
    
    # Calculate improvements
    f1_improvement = ((ppo_f1 - cnn_f1) / cnn_f1 * 100) if cnn_f1 > 0 else 0
    accuracy_improvement = ((ppo_accuracy - cnn_accuracy) / cnn_accuracy * 100) if cnn_accuracy > 0 else 0
    
    print("\n" + "=" * 60)
    print("📈 Performance Comparison:")
    print("=" * 60)
    
    print("\n🧠 CNN Baseline:")
    print(f"  Accuracy:  {cnn_accuracy:.4f}")
    print(f"  Precision: {cnn_precision:.4f}")
    print(f"  Recall:    {cnn_recall:.4f}")
    print(f"  F1-Score:  {cnn_f1:.4f}")
    
    print("\n🤖 PPO Refined:")
    print(f"  Accuracy:  {ppo_accuracy:.4f}")
    print(f"  Precision: {ppo_precision:.4f}")
    print(f"  Recall:    {ppo_recall:.4f}")
    print(f"  F1-Score:  {ppo_f1:.4f}")
    
    print("\n🎯 Improvements:")
    print(f"  F1-Score:  {f1_improvement:+.2f}%")
    print(f"  Accuracy:  {accuracy_improvement:+.2f}%")
    print(f"  Precision: {ppo_precision - cnn_precision:+.4f}")
    print(f"  Recall:    {ppo_recall - cnn_recall:+.4f}")
    
    # Save results
    results = {
        "algorithm": "PPO",
        "training_time_seconds": training_time,
        "total_timesteps": total_timesteps,
        "cnn_baseline": {
            "accuracy": float(cnn_accuracy),
            "precision": float(cnn_precision),
            "recall": float(cnn_recall),
            "f1_score": float(cnn_f1)
        },
        "ppo_refined": {
            "accuracy": float(ppo_accuracy),
            "precision": float(ppo_precision),
            "recall": float(ppo_recall),
            "f1_score": float(ppo_f1)
        },
        "improvements": {
            "f1_score_percent": float(f1_improvement),
            "accuracy_percent": float(accuracy_improvement),
            "precision_delta": float(ppo_precision - cnn_precision),
            "recall_delta": float(ppo_recall - cnn_recall)
        },
        "ppo_config": ppo_config,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save to JSON
    Path("results").mkdir(exist_ok=True)
    results_path = "results/ppo_training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")
    
    # Save refined mask
    refined_mask_path = "data/ppo_refined_cloud_mask.tif"
    with rasterio.open('data/ground_truth.tif') as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype='uint8')
        
        with rasterio.open(refined_mask_path, 'w', **profile) as dst:
            dst.write(rl_binary, 1)
    
    print(f"💾 Refined cloud mask saved to: {refined_mask_path}")
    
    print("\n" + "=" * 60)
    print("✅ PPO Training and Evaluation Complete!")
    print("=" * 60)
    
    return model, results

if __name__ == "__main__":
    model, results = train_ppo()
