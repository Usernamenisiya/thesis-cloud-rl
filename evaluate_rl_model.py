#!/usr/bin/env python3
"""
Updated evaluation script for RL cloud mask refinement.
This should be used in Colab to replace the broken evaluation code.
"""

import numpy as np
from rl_environment import CloudMaskRefinementEnv
from cnn_inference import load_sentinel2_image, get_cloud_mask
import rasterio
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from stable_baselines3 import DQN

def load_ground_truth(gt_path):
    """Load ground truth mask."""
    with rasterio.open(gt_path) as src:
        gt = src.read(1).astype(np.uint8)
    return gt

def evaluate_rl_model(model_path="rl_cloud_refinement_model"):
    """Evaluate trained RL model and compare with CNN baseline."""
    print("🔍 Evaluating RL model performance...")

    # Load data
    image = load_sentinel2_image("data/sentinel2_image.tif")
    cnn_prob = get_cloud_mask(image)
    ground_truth = load_ground_truth("data/ground_truth.tif")

    # Convert ground truth to binary
    gt_binary = (ground_truth > 0).astype(np.uint8)

    # CNN baseline evaluation
    cnn_binary = (cnn_prob > 0.5).astype(np.uint8)
    accuracy = accuracy_score(gt_binary.flatten(), cnn_binary.flatten())
    precision = precision_score(gt_binary.flatten(), cnn_binary.flatten(), zero_division=0)
    recall = recall_score(gt_binary.flatten(), cnn_binary.flatten(), zero_division=0)
    f1 = f1_score(gt_binary.flatten(), cnn_binary.flatten(), zero_division=0)

    print("📊 CNN Baseline Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    # Load trained RL model
    try:
        model = DQN.load(model_path)
        print(f"✅ Loaded RL model from: {model_path}")
    except:
        print(f"❌ Could not load model from {model_path}")
        return None

    # Create evaluation environment
    eval_env = CloudMaskRefinementEnv(image, cnn_prob, ground_truth, patch_size=64)

    # Collect predictions from trained agent
    rl_predictions = np.zeros_like(ground_truth, dtype=np.float32)

    obs = eval_env.reset()
    done = False
    step_count = 0

    print("🔄 Running RL evaluation...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)

        # Store prediction for entire patch
        i, j = eval_env.current_pos
        patch_size = eval_env.patch_size
        rl_predictions[i:i+patch_size, j:j+patch_size] = action

        obs, reward, done, info = eval_env.step(action)
        step_count += 1

        if step_count % 1000 == 0:
            print(f"Step {step_count}: Position ({i}, {j})")

    print(f"✅ Evaluation completed in {step_count} steps")

    # Calculate RL metrics
    rl_binary = (rl_predictions > 0.5).astype(np.uint8)
    rl_accuracy = accuracy_score(gt_binary.flatten(), rl_binary.flatten())
    rl_precision = precision_score(gt_binary.flatten(), rl_binary.flatten(), zero_division=0)
    rl_recall = recall_score(gt_binary.flatten(), rl_binary.flatten(), zero_division=0)
    rl_f1 = f1_score(gt_binary.flatten(), rl_binary.flatten(), zero_division=0)

    print("\n🤖 RL Refined Performance:")
    print(f"  Accuracy: {rl_accuracy:.4f}")
    print(f"  Precision: {rl_precision:.4f}")
    print(f"  Recall: {rl_recall:.4f}")
    print(f"  F1-Score: {rl_f1:.4f}")

    improvement = ((rl_f1 - f1) / f1) * 100 if f1 > 0 else 0
    print(f"\n🎯 F1-Score Improvement: {improvement:+.2f}%")

    # Check prediction coverage
    non_zero_pixels = np.sum(rl_binary > 0)
    total_pixels = rl_binary.size
    coverage = non_zero_pixels / total_pixels
    print(f"📈 Prediction Coverage: {coverage:.1%}")

    return {
        'cnn_baseline': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        },
        'rl_refined': {
            'accuracy': float(rl_accuracy),
            'precision': float(rl_precision),
            'recall': float(rl_recall),
            'f1_score': float(rl_f1)
        },
        'improvement': {
            'f1_improvement_percent': float(improvement)
        }
    }

if __name__ == "__main__":
    results = evaluate_rl_model()
    if results:
        import json
        with open("corrected_training_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("💾 Results saved to: corrected_training_results.json")