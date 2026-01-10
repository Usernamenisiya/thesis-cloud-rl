#!/usr/bin/env python3
"""
Simple evaluation script to test the RL environment fix.
This will help verify that the RL agent now properly predicts entire patches.
"""

import numpy as np
from rl_environment import CloudMaskRefinementEnv
from cnn_inference import load_sentinel2_image, get_cloud_mask
import rasterio
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_ground_truth(gt_path):
    """Load ground truth mask."""
    with rasterio.open(gt_path) as src:
        gt = src.read(1).astype(np.uint8)
    return gt

def evaluate_random_agent():
    """Test evaluation with a random agent to verify the environment works."""
    print("Testing RL environment with random agent...")

    # Load data
    image = load_sentinel2_image("data/sentinel2_image.tif")
    cnn_prob = get_cloud_mask(image)
    ground_truth = load_ground_truth("data/ground_truth.tif")

    # Create environment
    env = CloudMaskRefinementEnv(image, cnn_prob, ground_truth, patch_size=64)

    # Collect predictions with random agent
    rl_predictions = np.zeros_like(ground_truth, dtype=np.float32)

    obs = env.reset()
    done = False
    step_count = 0

    while not done:
        # Random action (0 or 1)
        action = np.random.randint(0, 2)

        # Store prediction for entire patch
        i, j = env.current_pos
        patch_size = env.patch_size
        rl_predictions[i:i+patch_size, j:j+patch_size] = action

        obs, reward, done, info = env.step(action)
        step_count += 1

        if step_count % 100 == 0:
            print(f"Step {step_count}: Current position ({i}, {j}), Reward: {reward:.3f}")

    print(f"Evaluation completed in {step_count} steps")

    # Convert ground truth to binary (assuming cloud=1, clear=0)
    gt_binary = (ground_truth > 0).astype(np.uint8)

    # Calculate metrics
    rl_binary = (rl_predictions > 0.5).astype(np.uint8)
    accuracy = accuracy_score(gt_binary.flatten(), rl_binary.flatten())
    precision = precision_score(gt_binary.flatten(), rl_binary.flatten(), zero_division=0)
    recall = recall_score(gt_binary.flatten(), rl_binary.flatten(), zero_division=0)
    f1 = f1_score(gt_binary.flatten(), rl_binary.flatten(), zero_division=0)

    print("\nRandom Agent Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    # Check if predictions are no longer sparse
    non_zero_pixels = np.sum(rl_binary > 0)
    total_pixels = rl_binary.size
    coverage = non_zero_pixels / total_pixels

    print(f"  Prediction coverage: {coverage:.1%} ({non_zero_pixels}/{total_pixels} pixels predicted as cloud)")

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    evaluate_random_agent()