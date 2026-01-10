# Cloud Masking with CNN and RL

This project aims to improve cloud masking accuracy, especially for thin clouds, by using a pre-trained CNN (s2cloudless) and refining its output with Reinforcement Learning (DQN).

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. **Data Preparation**:
   - **Automatic Download**: Run `python data_download.py` to download a sample Sentinel-2 image from the s2cloudless repo and create a dummy ground truth mask.
   - **Manual Download** (for real data):
     - Sentinel-2: Download L2A .zip from [ESA Samples](https://sentinel.esa.int/web/sentinel/sample-products), unzip, and stack bands B01-B12 into a single .tif using GDAL.
     - Ground Truth: Use [CloudSEN12](https://zenodo.org/record/4172871) or create manually in QGIS.
   - Place in `data/` folder.

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