#!/usr/bin/env python3
"""
Quick evaluation script for already-trained PPO model.
Usage: python evaluate_saved_model.py <model_path>
"""
import sys
import numpy as np
import rasterio
from stable_baselines3 import PPO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from cnn_inference import load_sentinel2_image, get_cloud_mask
from rl_environment import CloudMaskRefinementEnv

def evaluate_model(model_path):
    """Evaluate a saved PPO model."""
    
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Load data
    print("Loading data...")
    image = load_sentinel2_image('data/sentinel2_image.tif')
    cnn_prob = get_cloud_mask(image)
    
    with rasterio.open('data/ground_truth.tif') as src:
        ground_truth = src.read(1)
    
    # Create evaluation environment
    eval_env = CloudMaskRefinementEnv(image, cnn_prob, ground_truth, patch_size=64)
    rl_predictions = np.zeros_like(ground_truth, dtype=np.uint8)
    
    print(f"\nEvaluating on {len(eval_env.all_positions)} patches...")
    
    # Evaluate all patches
    num_patches = len(eval_env.all_positions)
    
    for patch_idx in range(num_patches):
        obs, _ = eval_env.reset()
        i, j = eval_env.current_pos
        patch_size = eval_env.patch_size
        
        action, _ = model.predict(obs, deterministic=True)
        rl_predictions[i:i+patch_size, j:j+patch_size] = action
        
        obs, reward, done, truncated, info = eval_env.step(action)
        
        if (patch_idx + 1) % 1000 == 0:
            print(f"  Progress: {patch_idx + 1}/{num_patches}")
    
    print(f"✅ Evaluation complete!\n")
    
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
    
    print("=" * 60)
    print("📈 RESULTS")
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
    
    f1_improvement = ((ppo_f1 - cnn_f1) / cnn_f1 * 100) if cnn_f1 > 0 else 0
    print(f"\n🎯 F1-Score Improvement: {f1_improvement:+.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_saved_model.py <model_path>")
        print("Example: python evaluate_saved_model.py models/ppo_cloud_refinement_model_20260112_062116")
        sys.exit(1)
    
    model_path = sys.argv[1]
    evaluate_model(model_path)
