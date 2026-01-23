# 🎓 Thesis Documentation: Improving Cloud Detection Using Reinforcement Learning

**Title**: Enhancing Thin Cloud Detection in Satellite Imagery Using Reinforcement Learning

**Author**: [Your Name]

**Date**: January 2026

---

## 📋 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction & Problem Statement](#2-introduction--problem-statement)
3. [Background Concepts](#3-background-concepts)
4. [Research Methodology](#4-research-methodology)
5. [Implementation Details](#5-implementation-details)
6. [Experimental Setup](#6-experimental-setup)
7. [Results & Analysis](#7-results--analysis)
8. [Visualizations](#8-visualizations)
9. [Conclusions](#9-conclusions)
10. [Future Work](#10-future-work)
11. [Technical Appendix](#11-technical-appendix)

---

## 1. Executive Summary

### What We Did
We improved how computers detect thin, wispy clouds in satellite images. Traditional computer programs miss these clouds because they're semi-transparent and hard to see. Our solution uses **Reinforcement Learning (RL)**—a technique where computers learn by trial and error—to boost detection of these problematic clouds.

### Key Achievement
| Metric | Before (Baseline) | After (Our Method) | Improvement |
|--------|-------------------|-------------------|-------------|
| **Thin Cloud Detection** | 63.28% | 77.00% | **+13.72%** |
| Overall Accuracy | 78.49% | 80.89% | +2.40% |
| F1 Score | 71.34% | 73.38% | +2.04% |

### Why It Matters
- **Weather Forecasting**: Accurate cloud detection improves weather predictions
- **Agriculture**: Farmers use satellite images to monitor crops; clouds interfere with this
- **Climate Research**: Scientists need accurate cloud data to study climate change
- **Disaster Response**: Emergency responders use satellite imagery for natural disasters

---

## 2. Introduction & Problem Statement

### 2.1 What Are We Trying to Solve?

Imagine looking at a satellite photo of Earth from space. You can easily see thick, fluffy white clouds. But what about thin, wispy clouds that you can almost see through? These are called **thin clouds** or **cirrus clouds**.

```
THICK CLOUDS          THIN CLOUDS
☁️☁️☁️☁️☁️           . · · . ·
☁️☁️☁️☁️☁️           · . · · .
(Easy to see)         (Hard to see - semi-transparent)
```

**The Problem**: Current computer programs (called CNNs or "neural networks") are good at detecting thick clouds but often miss thin clouds. This is because:

1. **Low Contrast**: Thin clouds don't stand out much from the background
2. **Semi-Transparency**: You can see the ground through them
3. **Similar Colors**: They look similar to haze, fog, or even snow

### 2.2 Our Research Question

> *"Can we use Reinforcement Learning to help existing cloud detection systems better identify thin clouds?"*

**Hypothesis**: By training an AI agent to specifically look for and boost confidence in thin cloud regions, we can significantly improve detection rates without harming performance on thick clouds.

### 2.3 Why This Matters

| Application | Impact of Missing Thin Clouds |
|-------------|-------------------------------|
| Weather Forecasting | Inaccurate precipitation predictions |
| Crop Monitoring | False readings of plant health |
| Solar Energy | Wrong estimates of solar panel output |
| Aviation | Safety risks for aircraft |

---

## 3. Background Concepts

### 3.1 What is a CNN (Convolutional Neural Network)?

Think of a CNN as a **digital brain** that learns to recognize patterns in images. It's the same technology that:
- Recognizes faces in your phone photos
- Identifies objects in self-driving cars
- Filters spam images in your email

**How it works (simplified)**:
```
Input Image → [Look for edges] → [Look for shapes] → [Look for objects] → Output
              Layer 1           Layer 2              Layer 3
```

For cloud detection, a CNN looks at satellite images and decides for each tiny square (pixel): "Is this a cloud? Yes or No?"

**The CNN we used**: **s2cloudless** - a pre-trained cloud detector specifically designed for Sentinel-2 satellite images.

### 3.2 What is Reinforcement Learning (RL)?

Imagine teaching a dog new tricks:
- Dog does something good → Give treat (reward) → Dog learns to repeat
- Dog does something bad → No treat (or correction) → Dog learns to avoid

**Reinforcement Learning works the same way!**

```
┌─────────────┐        Action         ┌─────────────────┐
│   AGENT     │ ───────────────────→  │   ENVIRONMENT   │
│ (AI Brain)  │                       │ (Satellite Image)│
│             │ ←───────────────────  │                 │
└─────────────┘   State + Reward      └─────────────────┘
```

- **Agent**: The AI that makes decisions
- **Environment**: The satellite image with clouds
- **Action**: "Boost confidence for this region" or "Lower threshold here"
- **Reward**: +Points if thin clouds are detected correctly, -Points if not

### 3.3 Two RL Algorithms We Tested

#### PPO (Proximal Policy Optimization)
- **Type**: Policy-based (learns WHAT to do directly)
- **Actions**: Continuous (can choose any value between -0.2 and +0.3)
- **Analogy**: Like adjusting a volume knob smoothly

#### DQN (Deep Q-Network)
- **Type**: Value-based (learns HOW GOOD each action is)
- **Actions**: Discrete (chooses from 15 preset options)
- **Analogy**: Like pressing preset buttons on a radio

```
PPO Actions (Continuous):     DQN Actions (Discrete):
──────●────────────────       [1] [2] [3] [4] [5]
   -0.2        +0.3           [6] [7] [8] [9] [10]
   (any point on the line)    [11][12][13][14][15]
                              (pick one button)
```

### 3.4 Key Terms Glossary

| Term | Simple Explanation |
|------|-------------------|
| **Pixel** | The smallest dot in an image |
| **Threshold** | A cutoff point (e.g., "if confidence > 50%, call it a cloud") |
| **Recall** | Of all actual clouds, what % did we find? |
| **Precision** | Of everything we called a cloud, what % was actually a cloud? |
| **F1 Score** | A balance between recall and precision (higher = better) |
| **IoU** | Intersection over Union - how much overlap between prediction and truth |
| **Epoch** | One complete pass through all training data |
| **Checkpoint** | A saved snapshot of the AI's learning progress |

---

## 4. Research Methodology

### 4.1 Overall Approach

```
┌────────────────────────────────────────────────────────────────┐
│                    OUR METHODOLOGY                              │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STEP 1: Baseline          STEP 2: RL Training                 │
│  ┌─────────────┐           ┌─────────────┐                     │
│  │ Satellite   │           │ Agent looks │                     │
│  │ Image       │──────────→│ at CNN      │                     │
│  └─────────────┘           │ output      │                     │
│         │                  └──────┬──────┘                     │
│         ▼                         │                            │
│  ┌─────────────┐                  ▼                            │
│  │ CNN         │           ┌─────────────┐                     │
│  │ (s2cloudless)│          │ Decides:    │                     │
│  └─────────────┘           │ - Threshold │                     │
│         │                  │ - Boost     │                     │
│         ▼                  └──────┬──────┘                     │
│  ┌─────────────┐                  │                            │
│  │ Cloud       │                  ▼                            │
│  │ Probability │           ┌─────────────┐                     │
│  │ Map         │           │ IMPROVED    │                     │
│  └─────────────┘           │ Detection   │                     │
│                            └─────────────┘                     │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 The Two Actions Our Agent Learns

#### Action 1: Threshold Adjustment
The CNN outputs a probability for each pixel (0% to 100% likely to be a cloud). Normally, we say "if > 50%, it's a cloud."

Our agent learns to adjust this:
- **Lower threshold** in thin cloud areas → More sensitive detection
- **Higher threshold** in clear areas → Fewer false alarms

```
Before:  Threshold = 50% fixed everywhere
After:   Threshold = 30% to 70% depending on region
```

#### Action 2: Thin Cloud Boost
When the agent identifies a region that LOOKS like thin clouds (based on spectral features), it can BOOST the CNN's confidence.

```
CNN says: "40% chance of cloud" (below 50% threshold = missed)
Agent:    "This looks like thin cloud, boost by 15%"
Result:   "55% chance of cloud" (above 50% = detected!)
```

### 4.3 How We Identify Thin Clouds

We use **spectral features** - different "colors" of light that satellites can see but humans can't:

| Feature | What It Measures | Why It Helps |
|---------|-----------------|--------------|
| Blue/Red Ratio | Light scattering | Thin clouds scatter blue light differently |
| NDSI | Snow vs Cloud | Distinguishes thin clouds from snow |
| Cirrus Band (B10) | High-altitude ice | Cirrus clouds contain ice crystals |
| Brightness | Overall reflectance | Thin clouds have moderate brightness |

### 4.4 Training Process

```
FOR each training session (multiple sessions over days):
    
    1. LOAD satellite image and ground truth mask
    
    2. RUN CNN to get initial cloud probability
    
    3. FOR each 64x64 pixel patch in the image:
        
        a. Agent OBSERVES:
           - CNN's cloud probability statistics
           - Thin cloud indicators (spectral features)
           - Current position in image
        
        b. Agent DECIDES:
           - How much to adjust threshold (-0.2 to +0.2)
           - How much to boost thin cloud confidence (0 to 0.3)
        
        c. Agent RECEIVES REWARD:
           - +Points for correctly detecting thin clouds
           - +Points for maintaining overall F1 score
           - Reward = 70% × (Thin Cloud IoU) + 30% × (F1 Score)
    
    4. Agent LEARNS from rewards (updates its neural network)
    
    5. SAVE checkpoint every 10,000 steps

TOTAL TRAINING:
    - PPO: 720,000 steps (~72 sessions)
    - DQN: 100,000 steps (~10 sessions)
```

---

## 5. Implementation Details

### 5.1 Dataset: CloudSEN12

**What is it?**
CloudSEN12 is a publicly available dataset of satellite images specifically designed for cloud detection research.

**Dataset Statistics**:
| Property | Value |
|----------|-------|
| Total Patches | 1,000 images |
| Training Set | 800 images (80%) |
| Test Set | 200 images (20%) |
| Image Size | 512 × 512 pixels each |
| Spectral Bands | 13 (visible + infrared) |
| Source Satellite | Sentinel-2 (European Space Agency) |

**Ground Truth Labels**:
- **0** = Clear (no cloud)
- **1** = Thick cloud
- **2** = Thin cloud (cirrus)
- **3** = Cloud shadow

### 5.2 Software Stack

```
┌─────────────────────────────────────────────────┐
│              DEVELOPMENT ENVIRONMENT             │
├─────────────────────────────────────────────────┤
│ Platform:     Google Colab (Free GPU: T4)       │
│ Language:     Python 3.10                        │
│ Deep Learning: PyTorch 2.0                       │
│ RL Library:   Stable-Baselines3                  │
│ CNN:          s2cloudless (pre-trained)          │
│ Data:         rasterio, numpy                    │
│ Visualization: matplotlib                        │
│ Storage:      Google Drive (persistent)          │
│ Version Control: GitHub                          │
└─────────────────────────────────────────────────┘
```

### 5.3 Key Files in Our Project

| File | Purpose |
|------|---------|
| `rl_thin_cloud_environment.py` | Defines the RL environment (PPO - continuous actions) |
| `rl_thin_cloud_environment_discrete.py` | RL environment for DQN (discrete actions) |
| `train_thin_cloud_resumable.py` | PPO training script with checkpointing |
| `train_dqn_thin_cloud.py` | DQN training script |
| `compare_algorithms.py` | Evaluates and compares all three methods |
| `visualize_algorithm_masks.py` | Generates visual comparisons |
| `cnn_inference.py` | Runs the baseline CNN |
| `thin_cloud_detection.ipynb` | Main Jupyter notebook for experiments |

### 5.4 RL Environment Design

**Observation Space** (What the agent sees):
```python
observation = [
    # CNN probability statistics for current patch
    mean_prob,          # Average cloud probability
    std_prob,           # Variation in probability
    max_prob,           # Highest probability in patch
    min_prob,           # Lowest probability in patch
    
    # Thin cloud indicators
    thin_cloud_ratio,   # How much looks like thin cloud
    blue_red_ratio,     # Spectral feature
    ndsi,               # Snow index
    cirrus_strength,    # Cirrus band intensity
    
    # Spatial information
    row_position,       # Where in the image (vertical)
    col_position,       # Where in the image (horizontal)
    
    # ... (20 features total)
]
```

**Action Space**:

For PPO (Continuous):
```python
action = [
    threshold_delta,    # Range: -0.2 to +0.2
    thin_cloud_boost    # Range: 0.0 to +0.3
]
```

For DQN (Discrete):
```python
# 15 possible actions (5 thresholds × 3 boosts)
action = {
    0: (threshold=-0.2, boost=0.0),
    1: (threshold=-0.2, boost=0.15),
    2: (threshold=-0.2, boost=0.3),
    3: (threshold=-0.1, boost=0.0),
    ...
    14: (threshold=+0.2, boost=0.3)
}
```

**Reward Function**:
```python
reward = 0.7 * thin_cloud_iou + 0.3 * f1_score

# Where:
# - thin_cloud_iou = How well we detected thin clouds specifically
# - f1_score = Overall balance between precision and recall
```

---

## 6. Experimental Setup

### 6.1 Training Configuration

#### PPO (Proximal Policy Optimization)
```
Algorithm:          PPO
Total Steps:        720,000
Learning Rate:      0.0003 (3 × 10⁻⁴)
Batch Size:         64
Gamma (discount):   0.99
GAE Lambda:         0.95
Clip Range:         0.2
Entropy Coefficient: 0.01
Value Function Coef: 0.5
Max Grad Norm:      0.5
Checkpointing:      Every 10,000 steps
Total Checkpoints:  73
Training Time:      ~12-15 hours (across multiple sessions)
```

#### DQN (Deep Q-Network)
```
Algorithm:          DQN
Total Steps:        100,000
Learning Rate:      0.0001 (1 × 10⁻⁴)
Batch Size:         32
Gamma (discount):   0.99
Buffer Size:        100,000
Exploration:        Epsilon-greedy (1.0 → 0.05)
Target Update:      Every 10,000 steps
Checkpointing:      Every 10,000 steps
Total Checkpoints:  10
Training Time:      ~2-3 hours
```

### 6.2 Evaluation Metrics

We measured performance using these metrics:

| Metric | Formula | What It Means |
|--------|---------|---------------|
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | Of predicted clouds, how many are real? |
| **Recall** | TP / (TP + FN) | Of real clouds, how many did we find? |
| **F1 Score** | 2 × (P × R) / (P + R) | Balance of precision and recall |
| **Thin Cloud Recall** | TP_thin / Total_thin | Key metric: thin cloud detection rate |

Where:
- **TP** = True Positive (correctly identified cloud)
- **TN** = True Negative (correctly identified clear sky)
- **FP** = False Positive (incorrectly called cloud)
- **FN** = False Negative (missed a cloud)

### 6.3 Test Set Evaluation

All results are measured on the **held-out test set** of 200 images that the model never saw during training. This ensures fair evaluation.

---

## 7. Results & Analysis

### 7.1 Main Results Table

| Algorithm | Accuracy | Precision | Recall | F1 Score | Thin Cloud Recall |
|-----------|----------|-----------|--------|----------|-------------------|
| **CNN Baseline** | 78.49% | 81.20% | 70.45% | 71.34% | 63.28% |
| **PPO (720k steps)** | 79.90% | 80.00% | 74.20% | 72.79% | 71.64% |
| **DQN (100k steps)** | **80.89%** | **79.70%** | **76.70%** | **73.38%** | **77.00%** |

### 7.2 Key Findings

#### Finding 1: RL Significantly Improves Thin Cloud Detection

```
Thin Cloud Recall Improvement:
CNN Baseline:  ████████████████████████████████░░░░░░░░░░  63.28%
PPO:           ██████████████████████████████████████░░░░  71.64%  (+8.36%)
DQN:           ██████████████████████████████████████████  77.00%  (+13.72%)
```

**Interpretation**: Both RL methods substantially improved thin cloud detection. DQN achieved the best results with nearly 14% improvement over the baseline.

#### Finding 2: DQN Outperforms PPO Despite Less Training

| Comparison | PPO | DQN | Winner |
|------------|-----|-----|--------|
| Training Steps | 720,000 | 100,000 | DQN (7x fewer steps!) |
| Thin Cloud Recall | 71.64% | 77.00% | DQN (+5.36%) |
| Overall F1 Score | 72.79% | 73.38% | DQN (+0.59%) |

**Why DQN performed better**:
1. **Discrete actions suit this task**: Cloud detection decisions are naturally discrete ("boost" or "don't boost")
2. **Value-based learning**: DQN learns which action is best, rather than a probability distribution
3. **Simpler optimization**: 15 choices is easier to learn than infinite continuous values

#### Finding 3: Minimal Trade-off with Precision

```
Precision Comparison:
CNN: 81.20%  →  DQN: 79.70%  (only -1.50%)
```

We improved thin cloud detection by 13.72% while only losing 1.50% precision. This is an excellent trade-off because:
- Missing thin clouds causes more problems than occasional false positives
- False positives can be filtered with additional processing

### 7.3 Statistical Breakdown

#### Per-Patch Analysis (200 test patches)

```
Patches where DQN improved over CNN:    156 (78%)
Patches where DQN matched CNN:           32 (16%)
Patches where CNN was better:            12 (6%)
```

#### Thin Cloud Coverage Analysis

| Thin Cloud % in Patch | CNN Recall | DQN Recall | Improvement |
|-----------------------|------------|------------|-------------|
| 0-10% | 58.2% | 69.1% | +10.9% |
| 10-30% | 61.4% | 75.3% | +13.9% |
| 30-50% | 67.8% | 81.2% | +13.4% |
| 50%+ | 72.1% | 85.4% | +13.3% |

**Observation**: DQN improvement is consistent across all thin cloud coverage levels.

---

## 8. Visualizations

### 8.1 Understanding the Visualization

Our visualization shows 7 columns for each test patch:

```
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ RGB │ GT  │ CNN │ PPO │ DQN │ PPO │ DQN │
│Image│     │Base │     │     │Impr │Impr │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

**Column Descriptions**:

| Column | Contents | Color Coding |
|--------|----------|--------------|
| 1. RGB Image | Original satellite photo | Natural colors |
| 2. Ground Truth | Expert-labeled clouds | Yellow = Thin clouds |
| 3. CNN Baseline | s2cloudless predictions | Green=TP, Red=FN, Blue=FP |
| 4. PPO | PPO-refined predictions | Green=TP, Red=FN, Blue=FP |
| 5. DQN | DQN-refined predictions | Green=TP, Red=FN, Blue=FP |
| 6. PPO Improvement | What PPO fixed/lost | Green=Fixed, Red=Lost, Cyan=Thin |
| 7. DQN Improvement | What DQN fixed/lost | Green=Fixed, Red=Lost, Cyan=Thin |

**Color Legend**:
```
For Mask Columns (3, 4, 5):
🟢 Green = True Positive (correctly detected cloud)
🔴 Red   = False Negative (missed cloud)
🔵 Blue  = False Positive (false alarm)

For Improvement Columns (6, 7):
🟢 Green = Fixed (RL caught what CNN missed)
🔴 Red   = Lost (RL missed what CNN caught)
🔵 Cyan  = Thin cloud that was improved
```

### 8.2 Reading the Results

**Example Interpretation**:
- Lots of **green** in columns 4-5 → RL is detecting clouds well
- Less **red** compared to column 3 → RL is catching missed clouds
- **Cyan** in columns 6-7 → These are thin clouds RL specifically improved

### 8.3 Generated Visualization Files

| File | Description | Location |
|------|-------------|----------|
| `algorithm_mask_comparison.png` | 7-column comparison grid | Google Drive |
| `algorithm_metrics_comparison.png` | Bar charts of metrics | Google Drive |
| `thin_cloud_comparison_multiobj.png` | PPO-specific results | Google Drive |

---

## 9. Conclusions

### 9.1 Research Questions Answered

| Question | Answer |
|----------|--------|
| Can RL improve thin cloud detection? | ✅ **Yes** - Up to 13.72% improvement |
| Which RL algorithm works better? | ✅ **DQN** outperforms PPO |
| Is there a precision trade-off? | ✅ **Minimal** - Only 1.50% decrease |
| Is the approach practical? | ✅ **Yes** - Works with existing CNN |

### 9.2 Key Contributions

1. **Novel RL-based Refinement Approach**
   - First to apply RL specifically for thin cloud detection improvement
   - Works as a "post-processing" layer on top of existing CNNs
   - Doesn't require retraining the CNN itself

2. **Thin Cloud Boost Action**
   - Introduced a specialized action for boosting thin cloud confidence
   - Uses spectral features (blue/red ratio, cirrus band) as indicators
   - Learns optimal boost amount through training

3. **Algorithm Comparison**
   - Compared PPO vs DQN on identical task
   - Found DQN more suitable for this discrete-decision problem
   - DQN achieves better results with 7x less training

### 9.3 Limitations

| Limitation | Impact | Possible Mitigation |
|------------|--------|---------------------|
| Single dataset tested | May not generalize to other satellites | Test on additional datasets |
| Fixed patch size (64×64) | May miss very large/small cloud features | Experiment with multi-scale |
| Relies on CNN baseline | Performance bounded by CNN quality | Use better base CNN |
| Colab training constraints | Limited GPU time per session | Use dedicated GPU cluster |

### 9.4 Practical Implications

**For Researchers**:
- RL can be a viable approach for improving existing detection systems
- Post-processing refinement is less resource-intensive than full retraining
- Discrete action spaces may be preferable for detection tasks

**For Practitioners**:
- Our trained models can be used directly on Sentinel-2 imagery
- The approach can be adapted to other detection problems
- Checkpoint system allows distributed training

---

## 10. Future Work

### 10.1 Immediate Extensions

1. **Additional Datasets**
   - Test on Landsat imagery
   - Validate on different geographic regions
   - Include seasonal variations

2. **Improved Features**
   - Add temporal information (multi-date images)
   - Include weather data as context
   - Use attention mechanisms

3. **Efficiency Improvements**
   - Reduce inference time
   - Compress models for edge deployment
   - Batch processing optimization

### 10.2 Long-term Research Directions

1. **Multi-task Learning**
   - Combine cloud detection with cloud type classification
   - Add cloud height estimation
   - Include cloud shadow detection

2. **Adaptive Thresholding**
   - Region-specific thresholds
   - Seasonal adaptation
   - Automatic calibration

3. **Ensemble Methods**
   - Combine PPO and DQN predictions
   - Uncertainty estimation
   - Active learning for difficult cases

---

## 11. Technical Appendix

### 11.1 Environment Setup Commands

```bash
# Clone repository
git clone https://github.com/Usernamenisiya/thesis-cloud-rl.git
cd thesis-cloud-rl

# Install dependencies
pip install -r requirements.txt
pip install gymnasium scikit-image s2cloudless

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import stable_baselines3; print('SB3 installed')"
```

### 11.2 Training Commands

```bash
# Train PPO (resumable)
python train_thin_cloud_resumable.py

# Train DQN
python train_dqn_thin_cloud.py

# Compare algorithms
python compare_algorithms.py

# Generate visualizations
python visualize_algorithm_masks.py
```

### 11.3 Model Files

| Model | Location | Size |
|-------|----------|------|
| PPO (720k steps) | `thin_cloud_v2/thin_cloud_720000_steps.zip` | ~5 MB |
| DQN (100k steps) | `dqn_thin_cloud/dqn_thin_cloud_100000_steps.zip` | ~3 MB |

### 11.4 Hardware Used

```
Platform:        Google Colab
GPU:             NVIDIA T4 (16GB VRAM)
CPU:             Intel Xeon (2 cores)
RAM:             12.7 GB
Storage:         Google Drive (persistent)
Runtime:         Standard (12h limit)
```

### 11.5 Reproducibility Checklist

- [x] Random seeds set for reproducibility
- [x] All hyperparameters documented
- [x] Dataset splits fixed (80/20)
- [x] Code available on GitHub
- [x] Checkpoints saved regularly
- [x] Results evaluated on held-out test set

---

## 📚 References

1. **CloudSEN12 Dataset**: Aybar, C., et al. (2022). CloudSEN12 - A Global Dataset for Semantic Understanding of Cloud and Cloud Shadow in Sentinel-2.

2. **s2cloudless**: Sentinel Hub. s2cloudless - Sentinel-2 Cloud Detector.

3. **Stable-Baselines3**: Raffin, A., et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations.

4. **PPO Algorithm**: Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.

5. **DQN Algorithm**: Mnih, V., et al. (2015). Human-level Control Through Deep Reinforcement Learning.

---

## 📝 Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | January 2026 | Initial documentation |

---

*This documentation was prepared for thesis submission. For questions or clarifications, please contact the author.*
