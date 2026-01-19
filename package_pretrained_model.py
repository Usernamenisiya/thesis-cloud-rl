"""
Save the trained RL model as a reusable pretrained model.
This creates a clean package with the best checkpoint and documentation.

RUN THIS IN COLAB where the checkpoints are located!
"""

import os
import shutil
import glob
import json
from datetime import datetime

# Find the best checkpoint (240k steps - our best performing model)
checkpoint_dir = 'checkpoints/thin_cloud'
checkpoint_paths = sorted(glob.glob(f'{checkpoint_dir}/thin_cloud_*_steps.zip'))

if not checkpoint_paths:
    print("❌ No checkpoints found!")
    exit(1)

# Prefer 240k checkpoint (best performance based on evaluation)
best_checkpoint = None
for cp in checkpoint_paths:
    if '240000' in cp:
        best_checkpoint = cp
        break

# If 240k not found, use latest checkpoint >= 200k (peak performance range)
if not best_checkpoint:
    high_step_checkpoints = [cp for cp in checkpoint_paths if any(str(s) in cp for s in ['200000', '210000', '220000', '230000', '240000'])]
    if high_step_checkpoints:
        best_checkpoint = high_step_checkpoints[-1]
    else:
        best_checkpoint = checkpoint_paths[-1]
        print(f"⚠️  Warning: Using {os.path.basename(best_checkpoint)} (240k checkpoint not found)")

print(f"📦 Packaging model from: {best_checkpoint}")

# Create pretrained model directory
pretrained_dir = 'pretrained_models'
model_name = 'thin_cloud_rl_v1'
model_dir = f'{pretrained_dir}/{model_name}'

os.makedirs(model_dir, exist_ok=True)

# Copy the checkpoint
model_path = f'{model_dir}/model.zip'
shutil.copy(best_checkpoint, model_path)
print(f"✅ Model saved to: {model_path}")

# Create model card with metadata
model_card = {
    "model_name": "Thin Cloud Detection RL Agent v1.0",
    "version": "1.0.0",
    "date_created": datetime.now().isoformat(),
    "description": "Reinforcement learning agent trained to improve thin cloud detection by refining CNN outputs",
    
    "architecture": {
        "algorithm": "PPO (Proximal Policy Optimization)",
        "framework": "Stable-Baselines3",
        "policy": "MlpPolicy",
        "observation_space": {
            "type": "Box",
            "shape": "(20,)",
            "features": [
                "CNN probability statistics (mean, std, min, max, median)",
                "Thin cloud indicators",
                "Spatial statistics",
                "Spectral features"
            ]
        },
        "action_space": {
            "type": "Box", 
            "shape": "(2,)",
            "actions": [
                "threshold_delta: [-0.2, 0.2] - Threshold adjustment",
                "thin_cloud_boost: [0.0, 0.3] - Probability boost for thin clouds"
            ]
        }
    },
    
    "training": {
        "dataset": "CloudSEN12 (1000 patches)",
        "train_split": "800 patches (80%)",
        "test_split": "200 patches (20%)",
        "total_steps": 240000,
        "training_time": "~2-3 hours on GPU",
        "patch_size": "64x64 pixels",
        "patches_per_scene": 100,
        "reward": "IoU improvement on thin clouds only",
        "learning_rate": 0.0003,
        "batch_size": 64
    },
    
    "performance": {
        "overall": {
            "accuracy": 78.34,
            "precision": 88.54,
            "recall": 60.71,
            "f1_score": 72.03,
            "iou": 0.5629
        },
        "thin_cloud_specific": {
            "baseline_recall": 16.43,
            "rl_recall": 26.56,
            "absolute_improvement": 10.13,
            "relative_improvement": 61.7
        }
    },
    
    "usage": {
        "input": "Sentinel-2 image (H, W, 13 bands) + CNN cloud probability (H, W)",
        "output": "Refined cloud mask (H, W) with improved thin cloud detection",
        "preprocessing": "Images should be in reflectance (0-1 range) or raw DN (0-10000, will auto-normalize)",
        "inference_time": "~0.5 seconds per 512x512 image on GPU"
    },
    
    "limitations": [
        "Trained specifically on CloudSEN12 dataset",
        "Optimized for thin cloud detection (may not improve thick clouds)",
        "Requires CNN baseline (s2cloudless) for input features",
        "Fixed patch size (64x64) for action application"
    ],
    
    "citation": {
        "thesis": "Improving Thin Cloud Detection using Reinforcement Learning",
        "author": "Your Name",
        "year": 2026,
        "repository": "https://github.com/Usernamenisiya/thesis-cloud-rl"
    }
}

# Save model card
with open(f'{model_dir}/model_card.json', 'w') as f:
    json.dump(model_card, f, indent=2)

print(f"✅ Model card saved to: {model_dir}/model_card.json")

# Create README for the model
readme_content = f"""# Thin Cloud Detection RL Agent v1.0

**Pretrained reinforcement learning model for improving thin cloud detection in satellite imagery.**

## Model Description

This model is trained to refine CNN-based cloud detection outputs, specifically improving the detection of thin/cirrus clouds which are typically missed by traditional CNN approaches.

### Key Features
- 61.7% improvement in thin cloud recall
- Maintains overall detection quality (F1: 72.03%)
- Fast inference (~0.5s per 512×512 image)
- Works with any CNN cloud detector (tested with s2cloudless)

## Quick Start

### Installation
```bash
pip install stable-baselines3 rasterio numpy
```

### Basic Usage
```python
from stable_baselines3 import PPO
from cnn_inference import load_sentinel2_image, get_cloud_mask
from rl_thin_cloud_environment import ThinCloudDetectionEnv
import numpy as np

# Load the pretrained model
model = PPO.load('pretrained_models/{model_name}/model')

# Load your satellite image
image = load_sentinel2_image('your_image.tif')  # (H, W, bands)

# Get CNN baseline prediction
cnn_prob = get_cloud_mask(image)  # (H, W) probabilities

# Apply RL refinement
gt_dummy = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)  # Not needed for inference
env = ThinCloudDetectionEnv(image, cnn_prob, gt_dummy, patch_size=64)

refined_mask = np.zeros_like(cnn_prob, dtype=np.uint8)
obs, _ = env.reset()

for patch_idx in range(env.num_patches):
    action, _ = model.predict(obs, deterministic=True)
    
    i, j = env.current_pos
    ps = env.patch_size
    
    threshold_delta = np.clip(action[0], -0.2, 0.2)
    thin_boost = np.clip(action[1], 0.0, 0.3)
    
    cnn_patch = cnn_prob[i:i+ps, j:j+ps].copy()
    thin_indicator = env.thin_cloud_indicator[i:i+ps, j:j+ps]
    
    boosted_prob = np.clip(cnn_patch + thin_indicator * thin_boost, 0, 1)
    refined_mask[i:i+ps, j:j+ps] = (boosted_prob > (0.5 + threshold_delta)).astype(np.uint8)
    
    obs, _, done, _, _ = env.step(action)
    if done:
        break

# refined_mask now contains improved cloud detection
```

## Performance

### Overall Cloud Detection
| Metric | Baseline CNN | RL Refined | Improvement |
|--------|--------------|------------|-------------|
| Accuracy | 78.49% | 78.34% | -0.14% |
| Precision | 90.47% | 88.54% | -1.93% |
| Recall | 59.42% | 60.71% | **+1.29%** |
| F1 Score | 71.73% | 72.03% | **+0.30%** |
| IoU | 0.5592 | 0.5629 | **+0.37%** |

### Thin Cloud Detection (Primary Contribution)
| Metric | Baseline CNN | RL Model | Improvement |
|--------|--------------|----------|-------------|
| Thin Cloud Recall | 16.43% | 26.56% | **+10.13%** (61.7% relative) |

## Model Details

- **Algorithm:** PPO (Proximal Policy Optimization)
- **Framework:** Stable-Baselines3 (PyTorch)
- **Training Data:** CloudSEN12 (800 patches)
- **Test Data:** CloudSEN12 (200 patches)
- **Training Steps:** 240,000
- **Observation Space:** 20 compact features from CNN output
- **Action Space:** 2 continuous actions (threshold adjustment + thin cloud boost)

## Input Requirements

1. **Satellite Image:** 
   - Format: (Height, Width, 13 bands) for Sentinel-2
   - Values: Reflectance (0-1) or raw DN (0-10000, auto-normalized)

2. **CNN Cloud Probability:**
   - Format: (Height, Width)
   - Values: 0-1 probability map from any CNN detector

## Limitations

- Trained on CloudSEN12 dataset only
- Best suited for Sentinel-2 imagery
- Requires CNN baseline for feature extraction
- May not generalize to very different geographic regions
- Patch-based processing (64×64) may create edge artifacts

## Citation

If you use this model in your research, please cite:

```bibtex
@mastersthesis{{yourname2026,
  title={{Improving Thin Cloud Detection using Reinforcement Learning}},
  author={{Your Name}},
  year={{2026}},
  school={{Your University}}
}}
```

## License

This model is released for research and educational purposes.

## Contact

For questions or issues, please open an issue on [GitHub](https://github.com/Usernamenisiya/thesis-cloud-rl).

---

**Model Version:** 1.0.0  
**Release Date:** {datetime.now().strftime("%Y-%m-%d")}  
**Model Size:** ~8MB  
**Framework:** Stable-Baselines3 v2.x
"""

with open(f'{model_dir}/README.md', 'w') as f:
    f.write(readme_content)

print(f"✅ README saved to: {model_dir}/README.md")

print("\n" + "="*60)
print("✅ PRETRAINED MODEL PACKAGE COMPLETE")
print("="*60)
print(f"\nModel location: {model_dir}/")
print(f"Files created:")
print(f"  - model.zip (trained weights)")
print(f"  - model_card.json (metadata)")
print(f"  - README.md (documentation)")
print("\nYou can now share this model or use it in other projects!")
print("="*60)
