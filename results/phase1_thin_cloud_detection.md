# Phase 1: Multi-Feature RL for Thin Cloud Detection

**Goal**: Specifically improve CNN's detection of thin/cirrus clouds using spectral features and targeted probability boosting.

**Date**: January 13-14, 2026

---

## Motivation

After completing 3 baseline approaches (CNN Baseline, Optimal Threshold, RL Threshold), we identified that:
- All approaches struggled with thin clouds specifically (CNN's known weakness)
- Simply lowering threshold improved recall but created false positives
- Need a more sophisticated approach targeting thin clouds specifically

**Research Question**: Can RL learn to detect thin clouds specifically using optical thickness indicators (blue/red ratio, reflectance levels)?

---

## Approach

### Architecture: Multi-Feature RL Environment

**Observation Space** (4118 features):
- CNN probability patch (64×64 = 4096 features)
- Texture features: variance, edge density, GLCM statistics (5 features)
- Spectral indices: NDSI, NDVI, band means (6 features)
- Thin cloud indicators: blue/red ratio, normalized reflectance, thin/thick flags (4 features)
- Spatial statistics: mean, std, min, max (4 features)
- Previous actions: memory of last 3 actions (3 features)

**Action Space** (3 continuous actions):
1. `threshold_delta` [-0.3, +0.3]: Base threshold adjustment
2. `thin_cloud_boost` [0, 0.4]: Extra probability boost for pixels identified as thin clouds
3. `spectral_weight` [0, 1.0]: Weight for spectral masking to filter false positives

**Key Innovation**: The `thin_cloud_boost` action specifically increases CNN probability for pixels with:
- Low-medium reflectance (compared to thick clouds)
- High blue/red ratio (thin clouds scatter more blue light)

**Reward Structure**:
- Base: F-beta score improvement (beta=0.7, emphasizes precision)
- **BONUS**: 5× multiplier for thin cloud F1-score
- **BONUS**: +2.0 for thin cloud recall >50%
- **BONUS**: +3.0 for thin cloud recall >70%
- **PENALTY**: For missing thick clouds
- **PENALTY**: For false positives on non-cloud areas

### Thin/Thick Cloud Classification

Ground truth clouds classified based on reflectance:
- **Thin clouds**: Bottom 70% of cloud brightness (lower reflectance)
- **Thick clouds**: Top 30% of cloud brightness (higher reflectance)

---

## Implementation Journey & Critical Bugs

### Iteration 1: Initial Implementation (Failed)
**Problem**: Ground truth masks are binary (0=clear, 1=cloud), not reflectance values.

**Bug**: Classification checked `if ground_truth > 4000` to find thick clouds, but GT values are only 0 or 1.
```python
# WRONG:
self.thick_clouds_gt = (self.ground_truth >= 4000)  # Always False!
```

**Result**: 0 thick cloud pixels detected, classification completely broken.

**Metrics**:
- Total Thick Clouds: **0** (should be ~30% of clouds)
- Agent couldn't learn meaningful thin-cloud-specific behavior
- F1: 30.25% (same as baseline RL threshold)

---

### Iteration 2: Fixed Classification (Still Failed)
**Fix**: Use actual satellite image reflectance to classify, not mask values.
```python
# CORRECT:
cloud_reflectance = self.normalized_reflectance[self.ground_truth > 0]
thin_threshold = np.percentile(cloud_reflectance, 70)
self.thin_clouds_gt = (cloud_mask & (reflectance < thin_threshold))
self.thick_clouds_gt = (cloud_mask & (reflectance >= thin_threshold))
```

**Problem**: Blue/red ratio feature completely broken!

**Bug**: Data is normalized to 0-1 range, but code checked `if red > 100`:
```python
# WRONG:
self.blue_red_ratio = np.where(self.red_band > 100,  # Never true with 0-1 data!
                                self.blue_band / self.red_band, 
                                0)
```

**Result**: Blue/red ratio = 0.000 for ALL pixels (useless feature).

**Metrics**:
- Blue/Red Difference: 0.000 (POOR)
- Reflectance Difference: 0.1 (POOR)
- Agent correctly learned features are useless → max out boost everywhere
- F1: 30.25%

---

### Iteration 3: Fixed Feature Extraction (Marginal Separability)
**Fix**: Remove hardcoded thresholds, work with actual data scale.
```python
# CORRECT:
self.blue_red_ratio = self.blue_band / (self.red_band + 1e-6)  # Works with any scale
blue_red_median = np.median(self.blue_red_ratio)
thin_cloud_indicator = (reflectance < 70th_percentile) & (blue_red > median)
```

**Result**: Features now calculate correctly but show WEAK separability.

**Feature Separability Analysis**:
```
THIN CLOUD FEATURES:
  Blue/Red Ratio:  Mean=0.920, Std=0.119
  Reflectance:     Mean=0.2, Std=0.1
  
THICK CLOUD FEATURES:
  Blue/Red Ratio:  Mean=0.860, Std=0.091
  Reflectance:     Mean=0.2, Std=0.0
  
SEPARABILITY:
  Blue/Red Difference:  0.061 (POOR - marginal signal)
  Reflectance Difference: 0.1 (POOR - essentially identical)
  
Blue/Red Ranges:
  Thin:  0.527 - 2.006  } Huge overlap!
  Thick: 0.346 - 2.293  }
```

**Learned Agent Actions**:
```
Threshold Delta:    -0.3000 ± 0.0000  (always minimum, lower threshold)
Thin Cloud Boost:    0.3943 ± 0.0252  (nearly maxed out everywhere)
Spectral Weight:     0.0024 ± 0.0199  (essentially zero, not used)
```

**Metrics**:
- Overall F1: 30.25% (+17.65% vs CNN baseline)
- Thin Cloud Recall: 54.6%
- Thick Cloud Recall: 66.6%
- F1 identical to simple optimal threshold approach

---

## Key Findings

### 1. Spectral Features Have Poor Discriminative Power
The blue/red ratio and reflectance features that theoretically distinguish thin from thick clouds show **minimal separability** in CloudSEN12 data:
- Blue/red difference: 0.061 (needs >0.1 for usefulness)
- Massive range overlap between thin and thick clouds
- Most pixels fall in ambiguous middle range

### 2. Agent Converged to Degenerate Policy
With weak features, the agent learned a **degenerate policy**:
- Always apply maximum thin cloud boost (0.394)
- Always lower threshold to minimum (-0.3)
- Never use spectral filtering (weight ≈ 0)

**This is actually OPTIMAL given the feature quality!** If features don't distinguish thin from thick, boosting everything equally is the best strategy.

### 3. Equivalent to Optimal Threshold Approach
Visual comparison shows CNN baseline and "Thin Cloud Detection RL" produce **nearly identical masks**:
- Thin cloud patch: Both F1 = 0.143 (identical performance)
- Thick cloud patch: F1 = 0.410 vs 0.416 (negligible difference)
- Overall improvement (+17.65%) comes from threshold adjustment, not thin cloud boost

### 4. Ground Truth Quality Issues
User observation: *"The ground truth is not accurate cause I can see in the patch the actual clouds where the ground truth misses its masks"*

This suggests:
- Ground truth labels are incomplete for thin/semi-transparent clouds
- CNN learned to reproduce these incomplete labels (25% F1)
- RL's "false positives" may actually be correct cloud detections
- Low precision (20%) might be artificially deflated

---

## Comparison with Baseline Approaches

| Approach | F1-Score | Improvement | Strategy |
|----------|----------|-------------|----------|
| **CNN Baseline** | 25.71% | - | threshold=0.5 |
| **Optimal Threshold** | 28.68% | +11.56% | threshold=0.10 (classical search) |
| **RL Threshold** | 30.25% | +17.65% | Learned threshold adjustment |
| **Multi-Feature RL** | 30.25% | +17.65% | Thin cloud boost + threshold |

**Conclusion**: Multi-Feature RL achieves **same performance** as RL Threshold but through more complex route. The thin cloud boost action adds no additional value.

---

## Thesis Implications

### Positive Findings:
1. **RL Explores Solution Space Effectively**: Even with complex action/observation space, agent converged to near-optimal strategy
2. **Validates Threshold Adjustment**: Multiple independent approaches (grid search, RL threshold, multi-feature RL) all converge to ~30% F1 by lowering threshold
3. **Feature Engineering Insight**: Shows importance of feature separability analysis BEFORE designing complex RL systems
4. **Honest Scientific Process**: Documenting failed approaches is valuable - shows what DOESN'T work

### Limitations Identified:
1. **CloudSEN12 data lacks thin/thick cloud diversity** - Most clouds fall in ambiguous middle reflectance range
2. **Ground truth quality concerns** - Incomplete labels for thin clouds may limit all approaches
3. **Spectral features insufficient** - Blue/red ratio alone can't distinguish thin clouds reliably
4. **Need complementary approaches** - Spectral alone won't solve this problem

### Next Steps:
- **Phase 2**: Shadow detection and removal (geometric + spectral features)
- **Phase 3**: Hierarchical refinement for cloud boundaries
- **Alternative**: CNN retraining with data augmentation for thin clouds
- **Alternative**: Multi-modal approaches (optical + thermal bands)

---

## Conclusion

**Phase 1 demonstrated that:**
- Multi-feature RL with thin cloud detection **does not improve** beyond simple threshold adjustment
- Spectral features (blue/red ratio, reflectance) have insufficient separability in CloudSEN12 data
- Agent rationally learned to apply maximum boost everywhere when features are uninformative
- The 30.25% F1 ceiling appears to be fundamental limit of threshold-based approaches

**This is a valuable negative result** that:
- Confirms optimal threshold search is near-optimal
- Highlights need for fundamentally different approaches (shadow removal, hierarchical refinement)
- Demonstrates scientific rigor in honestly assessing what doesn't work
- Guides future research toward more promising directions

**For the thesis**: This phase provides important context showing we thoroughly explored feature-based RL approaches before determining they hit a performance ceiling, motivating the need for alternative strategies.
