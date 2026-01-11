# Cloud Masking with CNN and RL

This project aims to improve cloud masking accuracy, especially for thin clouds, by using a pre-trained CNN (s2cloudless) and refining its output with Reinforcement Learning (DQN).

## Data Setup

### Option 1: Google Drive (Recommended)
For persistent storage across Colab sessions:

1. Create folder: `Colab_Data/thesis_cloud_rl/` in your Google Drive
2. Upload your data files there
3. Run the Google Drive mounting cell in `colab_training.ipynb`

**Benefits:** No re-uploading needed!

### Option 2: Cloud Storage Downloads
Use `setup_cloud_data.py` for automatic downloads:

```bash
python setup_cloud_data.py
```

Supports Dropbox, OneDrive, Google Drive, and direct URLs.

### Option 3: Manual Upload
Upload files directly to Colab (temporary, resets each session).

### Option 4: Automatic Dataset Download
Modify `data_download.py` to fetch from public datasets like CloudSEN12.

## Recent Updates

### RL Environment Fix (Jan 2026)
**Issue**: Previous RL evaluation only predicted center pixels of patches, leading to 0 precision/recall/F1 scores.

**Fix**: Updated `rl_environment.py` to:
- Reward based on entire patch accuracy (0-1 range)
- Return patch positions in `info` for proper evaluation
- Fill entire patches during prediction instead of just centers

**Testing**: Run `python test_evaluation.py` to verify the fix works.

**Corrected Evaluation**: Use `python evaluate_rl_model.py` for proper RL model evaluation.

## Usage

1. **Load Data**: Update paths in `train_dqn.py` for your Sentinel-2 image and ground truth mask.
   - Sentinel-2 image: .tif with 12 bands (B01-B12).
   - Ground truth: Binary .tif mask (0=not cloud, 1=cloud).
2. **Run CNN Inference**: The `cnn_inference.py` loads the image, selects 10 bands (B02-B12 except B09/B10), and generates probability mask.
3. **Train RL**: Run `train_dqn.py` to train DQN on patches.
4. **Evaluate**: Compare refined mask to ground truth using metrics like IoU.

## Feasibility in 1 Month

With data ready, CNN inference is quick. RL training on patches can be done in days with GPU. Total: Possible, but focus on data preparation and hyperparameter tuning.