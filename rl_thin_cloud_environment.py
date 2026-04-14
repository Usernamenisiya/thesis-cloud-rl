"""
Thin Cloud Detection RL Environment - IMPROVED VERSION

Aligned with best practices:
 CNN output as RL state (not raw image)
 Patch-level actions (64x64)
 Reward = IoU improvement on THIN CLOUDS ONLY
 Proper normalization handling
 Focused reward structure

Author: Thesis Implementation
Date: January 2026
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from scipy import ndimage


class ThinCloudDetectionEnv(gym.Env):
    """
    Simplified RL environment focused ONLY on thin cloud detection.
    
    Key Design Decisions:
    - State: CNN probability map + thin cloud indicators (NOT raw pixels)
    - Actions: Patch-level threshold adjustment + thin cloud boost
    - Reward: IoU improvement on thin clouds ONLY (as recommended)
    
    Observation Space:
        - CNN probability patch statistics (mean, std, percentiles)
        - Thin cloud indicators (blue/red ratio, reflectance level)
        - Previous action memory
        
    Action Space:
        - threshold_delta: [-0.2, +0.2] - Conservative adjustment
        - thin_cloud_boost: [0, 0.3] - Boost for thin cloud pixels
    """
    
    def __init__(self, image, cnn_prob, ground_truth, patch_size=64, baseline_threshold=0.5):
        super().__init__()
        
        # Handle both (H,W,C) and (C,H,W) formats
        if image.shape[0] <= 13 and len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))
        
        self.image = image
        self.cnn_prob = cnn_prob
        self.ground_truth = ground_truth
        self.patch_size = patch_size
        self.baseline_threshold = baseline_threshold
        
        # Auto-detect if data is normalized (0-1) or raw (0-10000+)
        self.data_max = self.image.max()
        self.is_normalized = self.data_max <= 2.0
        
        # Calculate patches
        self.h, self.w = cnn_prob.shape
        self.num_patches_h = self.h // patch_size
        self.num_patches_w = self.w // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Pre-compute thin cloud classification (THE KEY)
        self._compute_thin_cloud_classification()
        
        # Pre-compute baseline metrics for thin clouds only
        self._compute_baseline_metrics()
        
        # Action space: [threshold_delta, thin_cloud_boost]
        self.action_space = spaces.Box(
            low=np.array([-0.2, 0.0]),
            high=np.array([0.2, 0.3]),
            dtype=np.float32
        )
        
        # Observation space: Compact CNN + thin cloud features
        # 10 CNN stats + 4 thin cloud features + 2 action memory + 4 spatial context = 20
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(20,), dtype=np.float32
        )
        
        # State
        self.current_patch_idx = 0
        self.current_pos = (0, 0)
        self.current_actions = np.array([0.0, 0.0])
        
    def _normalize_reflectance(self, band):
        """Normalize reflectance to 0-1 range."""
        if self.is_normalized:
            return np.clip(band, 0, 1)
        else:
            return np.clip(band / 10000.0, 0, 1)
    
    def _compute_thin_cloud_classification(self):
        """
        Classify ground truth clouds into thin vs thick.
        
        Uses reflectance-based classification:
        - Thin clouds: Lower reflectance (semi-transparent)
        - Thick clouds: Higher reflectance (opaque)
        """
        # Extract normalized bands
        if self.image.shape[2] >= 8:
            blue = self._normalize_reflectance(self.image[:, :, 1])
            green = self._normalize_reflectance(self.image[:, :, 2])
            red = self._normalize_reflectance(self.image[:, :, 3])
            nir = self._normalize_reflectance(self.image[:, :, 7])
        else:
            # Fallback
            blue = green = red = nir = np.zeros_like(self.cnn_prob)
        
        self.blue = blue
        self.red = red
        
        # Compute normalized reflectance
        self.normalized_reflectance = (blue + green + red + nir) / 4.0
        
        # Blue/Red ratio (thin clouds scatter more blue)
        self.blue_red_ratio = blue / (red + 1e-6)
        
        # Classify cloud thickness based on ground truth + reflectance
        cloud_mask = self.ground_truth > 0
        
        if cloud_mask.sum() > 100:
            # Use cloud pixels to determine threshold
            cloud_reflectance = self.normalized_reflectance[cloud_mask]
            # Bottom 70% = thin, top 30% = thick
            thickness_threshold = np.percentile(cloud_reflectance, 70)
        else:
            thickness_threshold = 0.3  # Default for normalized data
        
        # Thin clouds: lower reflectance clouds
        self.thin_clouds_gt = np.logical_and(
            cloud_mask,
            self.normalized_reflectance < thickness_threshold
        ).astype(np.uint8)
        
        # Thick clouds: higher reflectance
        self.thick_clouds_gt = np.logical_and(
            cloud_mask,
            self.normalized_reflectance >= thickness_threshold
        ).astype(np.uint8)
        
        # Thin cloud indicator for boosting (based on spectral properties, not GT)
        blue_red_median = np.median(self.blue_red_ratio)
        refl_30th = np.percentile(self.normalized_reflectance, 30)
        refl_70th = np.percentile(self.normalized_reflectance, 70)
        
        self.thin_cloud_indicator = np.logical_and(
            np.logical_and(self.normalized_reflectance > refl_30th,
                          self.normalized_reflectance < refl_70th),
            self.blue_red_ratio > blue_red_median
        ).astype(np.float32)
        
    def _compute_baseline_metrics(self):
        """Compute baseline IoU for thin clouds."""
        baseline_pred = (self.cnn_prob > self.baseline_threshold).astype(np.uint8)
        
        # Overall baseline
        gt_flat = (self.ground_truth > 0).astype(np.uint8).flatten()
        baseline_flat = baseline_pred.flatten()
        
        if gt_flat.sum() > 0:
            self.baseline_iou = jaccard_score(gt_flat, baseline_flat, zero_division=0)
        else:
            self.baseline_iou = 0.0
        
        # Thin cloud baseline IoU (THE KEY METRIC)
        thin_gt_flat = self.thin_clouds_gt.flatten()
        if thin_gt_flat.sum() > 0:
            self.baseline_thin_iou = jaccard_score(thin_gt_flat, baseline_flat, zero_division=0)
        else:
            self.baseline_thin_iou = 0.0
    
    def _get_observation(self):
        """
        Construct COMPACT observation focused on CNN output and thin cloud indicators.
        
        NOT using raw pixels - using CNN probability statistics as recommended.
        """
        i, j = self.current_pos
        ps = self.patch_size
        
        # CNN probability patch statistics (10 features)
        cnn_patch = self.cnn_prob[i:i+ps, j:j+ps]
        cnn_stats = np.array([
            cnn_patch.mean(),
            cnn_patch.std(),
            np.percentile(cnn_patch, 10),
            np.percentile(cnn_patch, 25),
            np.percentile(cnn_patch, 50),
            np.percentile(cnn_patch, 75),
            np.percentile(cnn_patch, 90),
            (cnn_patch > 0.3).mean(),  # % above low threshold
            (cnn_patch > 0.5).mean(),  # % above mid threshold
            (cnn_patch > 0.7).mean(),  # % above high threshold
        ])
        
        # Thin cloud features (4 features)
        refl_patch = self.normalized_reflectance[i:i+ps, j:j+ps]
        br_patch = self.blue_red_ratio[i:i+ps, j:j+ps]
        thin_ind_patch = self.thin_cloud_indicator[i:i+ps, j:j+ps]
        
        thin_features = np.array([
            refl_patch.mean(),  # Mean reflectance (normalized)
            np.clip(br_patch.mean() / 2.0, 0, 1),  # Blue/red ratio (normalized)
            thin_ind_patch.mean(),  # Thin cloud indicator proportion
            (cnn_patch > 0.3).mean() * thin_ind_patch.mean(),  # Interaction
        ])
        
        # Spatial context (4 features) - where are we in the image?
        spatial = np.array([
            i / self.h,  # Vertical position
            j / self.w,  # Horizontal position
            self.current_patch_idx / self.num_patches,  # Progress
            float(thin_ind_patch.sum() > ps * ps * 0.1),  # Has significant thin cloud potential
        ])
        
        # Action memory (2 features)
        action_memory = np.clip(self.current_actions, 0, 1)
        
        obs = np.concatenate([cnn_stats, thin_features, spatial, action_memory])
        return obs.astype(np.float32)
    
    def _compute_reward(self, final_pred, i, j):
        """
        Multi-objective reward:
        - 70% weight on thin cloud improvement
        - 30% weight on overall F1 improvement
        - Penalty for precision loss (prevent over-detection)
        """
        ps = self.patch_size
        
        # Ground truth for this patch
        gt_patch = (self.ground_truth[i:i+ps, j:j+ps] > 0).astype(np.uint8)
        thin_gt_patch = self.thin_clouds_gt[i:i+ps, j:j+ps]
        
        # Baseline prediction
        baseline_pred = (self.cnn_prob[i:i+ps, j:j+ps] > self.baseline_threshold).astype(np.uint8)
        
        # Flatten for metrics
        gt_flat = gt_patch.flatten()
        pred_flat = final_pred.flatten()
        baseline_flat = baseline_pred.flatten()
        thin_gt_flat = thin_gt_patch.flatten()
        
        reward = 0.0
        
    
        # COMPONENT 1: Thin cloud IoU improvement (70% weight)

        if thin_gt_flat.sum() > 0:
            baseline_thin_iou = jaccard_score(thin_gt_flat, baseline_flat, zero_division=0)
            adjusted_thin_iou = jaccard_score(thin_gt_flat, pred_flat, zero_division=0)
            thin_improvement = adjusted_thin_iou - baseline_thin_iou
            reward += 0.7 * thin_improvement * 10.0
        
     
        # COMPONENT 2: Overall F1 improvement (30% weight)

        if gt_flat.sum() > 0:
            baseline_f1 = f1_score(gt_flat, baseline_flat, zero_division=0)
            adjusted_f1 = f1_score(gt_flat, pred_flat, zero_division=0)
            f1_improvement = adjusted_f1 - baseline_f1
            reward += 0.3 * f1_improvement * 10.0
        

        # PENALTY: Precision loss (prevent over-detection)

        if pred_flat.sum() > 0:
            baseline_precision = precision_score(gt_flat, baseline_flat, zero_division=1)
            adjusted_precision = precision_score(gt_flat, pred_flat, zero_division=1)
            precision_loss = baseline_precision - adjusted_precision
            
            if precision_loss > 0.1:  # Lost more than 10% precision
                reward -= precision_loss * 5.0  # Strong penalty
        
        # PENALTY: False positives on clear sky

        if gt_flat.sum() == 0 and pred_flat.sum() > 0:
            false_positive_rate = pred_flat.sum() / pred_flat.size
            reward -= false_positive_rate * 3.0
        
        return reward
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        self.current_patch_idx = 0
        self.current_pos = (0, 0)
        self.current_actions = np.array([0.0, 0.0])
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute action and compute thin-cloud-focused reward."""
        i, j = self.current_pos
        ps = self.patch_size
        
        # Clip actions
        threshold_delta = np.clip(action[0], -0.2, 0.2)
        thin_cloud_boost = np.clip(action[1], 0.0, 0.3)
        self.current_actions = np.array([
            (threshold_delta + 0.2) / 0.4,  # Normalize to 0-1
            thin_cloud_boost / 0.3  # Normalize to 0-1
        ])
        
        # Apply adjusted threshold
        adjusted_threshold = self.baseline_threshold + threshold_delta
        
        # Get CNN patch and apply thin cloud boost
        cnn_patch = self.cnn_prob[i:i+ps, j:j+ps].copy()
        
        # Boost thin-cloud-like pixels
        thin_indicator = self.thin_cloud_indicator[i:i+ps, j:j+ps]
        cnn_boosted = cnn_patch + thin_indicator * thin_cloud_boost
        cnn_boosted = np.clip(cnn_boosted, 0, 1)
        
        # Final prediction
        final_pred = (cnn_boosted > adjusted_threshold).astype(np.uint8)
        
        # Compute reward (thin cloud IoU focused)
        reward = self._compute_reward(final_pred, i, j)
        
        # Move to next patch
        self.current_patch_idx += 1
        done = self.current_patch_idx >= self.num_patches
        
        if not done:
            patch_row = self.current_patch_idx // self.num_patches_w
            patch_col = self.current_patch_idx % self.num_patches_w
            self.current_pos = (patch_row * ps, patch_col * ps)
        
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        
        return obs, reward, done, False, {}


def test_environment():
    """Quick test."""
    print("Testing ThinCloudDetectionEnv...")
    
    # Simulate normalized data (0-1 range)
    image = np.random.rand(512, 512, 10)
    cnn_prob = np.random.rand(512, 512)
    ground_truth = (np.random.rand(512, 512) > 0.7).astype(np.uint8)
    
    env = ThinCloudDetectionEnv(image, cnn_prob, ground_truth)
    
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Is normalized data: {env.is_normalized}")
    print(f"Baseline thin cloud IoU: {env.baseline_thin_iou:.4f}")
    
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    print(f"Reward: {reward:.4f}")
    
    print(" Test passed!")


if __name__ == "__main__":
    test_environment()
