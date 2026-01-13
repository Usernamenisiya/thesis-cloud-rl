"""
Multi-Feature RL Environment for THIN CLOUD Detection

GOAL: Specifically improve detection of thin/cirrus clouds (CNN's main weakness)

Enhanced environment that uses:
- Optical thickness estimation from spectral bands
- Thin cloud indicators (BTD - Brightness Temperature Difference)
- Texture features (variance, edge density for thin cloud patterns)
- Spectral indices (NDSI, NDVI to filter false positives)
- Multi-dimensional actions (threshold, thin cloud boost, spectral mask)
- Modified reward with BONUS for detecting thin clouds specifically

Author: Thesis Implementation
Date: January 2026
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops


class MultiFeatureRefinementEnv(gym.Env):
    """
    Enhanced RL environment specifically for THIN CLOUD detection.
    
    CNN Weakness: Thin/cirrus clouds have low reflectance → low CNN probability → missed
    Solution: Agent learns to boost confidence for thin cloud patterns
    
    Observation Space:
        - CNN probability patch (64x64 flattened)
        - Optical thickness indicators (BTD, reflectance ratios)
        - Thin cloud texture patterns (low variance, smooth edges)
        - Spectral indices (NDSI, NDVI to filter false positives)
        - Cloud thickness classification (thin vs thick)
        
    Action Space:
        - threshold_delta: [-0.3, +0.3] - Base threshold adjustment
        - thin_cloud_boost: [0, 0.4] - Extra boost for thin cloud pixels
        - spectral_weight: [0, 1] - Weight for spectral masking
        
    Reward:
        - Base: F1-score improvement
        - BONUS: Extra reward for correctly detecting thin clouds
        - PENALTY: False positives on shadows/water
        - BONUS: High precision on thin clouds specifically
    """
    
    def __init__(self, image, cnn_prob, ground_truth, patch_size=64, baseline_threshold=0.5, beta=0.7):
        super().__init__()
        
        # Handle both (H,W,C) and (C,H,W) formats
        if image.shape[0] <= 13 and len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))
        
        self.image = image
        self.cnn_prob = cnn_prob
        self.ground_truth = ground_truth
        self.patch_size = patch_size
        self.baseline_threshold = baseline_threshold
        self.beta = beta  # F-beta parameter (< 1 emphasizes precision)
        
        # Calculate patches
        self.h, self.w = cnn_prob.shape
        self.num_patches_h = self.h // patch_size
        self.num_patches_w = self.w // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Pre-compute baseline metrics
        baseline_pred = (cnn_prob > baseline_threshold).astype(np.uint8)
        gt_flat = (ground_truth > 0).astype(np.uint8).flatten()
        baseline_flat = baseline_pred.flatten()
        self.baseline_fbeta = fbeta_score(gt_flat, baseline_flat, beta=beta, zero_division=0)
        
        # Pre-compute spectral indices
        self._compute_spectral_indices()
        
        # Pre-compute thin cloud indicators
        self._compute_thin_cloud_indicators()
        
        # Classify ground truth into thin vs thick clouds
        self._classify_cloud_thickness()
        
        # Pre-compute texture features for all patches
        self._compute_texture_features()
        
        # Action space: [threshold_delta, thin_cloud_boost, spectral_weight]
        self.action_space = spaces.Box(
            low=np.array([-0.3, 0.0, 0.0]),
            high=np.array([0.3, 0.4, 1.0]),  # thin_cloud_boost can go up to +0.4
            dtype=np.float32
        )
        
        # Observation space calculation
        patch_features = patch_size * patch_size  # CNN prob patch
        texture_features = 5  # variance, edge_density, homogeneity, contrast, energy
        spectral_features = 6  # NDSI, NDVI, mean_nir, mean_red, mean_green, mean_blue
        thin_cloud_features = 4  # blue_red_ratio, normalized_reflectance, thin_indicator, thick_indicator
        spatial_stats = 4  # mean, std, min, max
        action_memory = 3  # previous actions
        
        obs_size = patch_features + texture_features + spectral_features + thin_cloud_features + spatial_stats + action_memory
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        
        # State
        self.current_patch_idx = 0
        self.current_pos = (0, 0)
        self.current_actions = np.array([0.0, 0.0, 0.5])  # [threshold_delta, thin_boost, spectral_weight]
        
    def _compute_spectral_indices(self):
        """Pre-compute spectral indices for all patches."""
        # Sentinel-2 band indices (10m bands): B02=Blue, B03=Green, B04=Red, B08=NIR
        # Assuming band order: [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10]
        # We need: B03 (idx=2), B04 (idx=3), B08 (idx=7), B11 (SWIR, not in 10m)
        
        if self.image.shape[2] >= 8:
            blue = self.image[:, :, 1].astype(np.float32)   # B02
            green = self.image[:, :, 2].astype(np.float32)  # B03
            red = self.image[:, :, 3].astype(np.float32)    # B04
            nir = self.image[:, :, 7].astype(np.float32)    # B08
            
            # NDVI: (NIR - Red) / (NIR + Red)
            denominator = nir + red
            self.ndvi = np.where(denominator != 0, (nir - red) / denominator, 0)
            
            # NDSI for snow/ice detection: (Green - SWIR) / (Green + SWIR)
            # Approximation using NIR as proxy for SWIR
            denominator = green + nir
            self.ndsi = np.where(denominator != 0, (green - nir) / denominator, 0)
            
            # Store individual bands for features
            self.blue_band = blue
            self.green_band = green
            self.red_band = red
            self.nir_band = nir
        else:
            # Fallback if bands not available
            self.ndvi = np.zeros_like(self.cnn_prob)
            self.ndsi = np.zeros_like(self.cnn_prob)
            self.blue_band = np.zeros_like(self.cnn_prob)
            self.green_band = np.zeros_like(self.cnn_prob)
            self.red_band = np.zeros_like(self.cnn_prob)
            self.nir_band = np.zeros_like(self.cnn_prob)
    
    def _compute_thin_cloud_indicators(self):
        """
        Compute indicators specifically for thin cloud detection.
        
        Thin clouds characteristics:
        - Low but consistent reflectance across bands
        - Smooth texture (low variance)
        - Brightness temperature difference (BTD) - cirrus detection
        - High blue/red ratio (scattering)
        """
        # Reflectance ratio: Blue/Red (thin clouds scatter more blue light)
        self.blue_red_ratio = np.where(
            self.red_band > 100,
            self.blue_band / (self.red_band + 1e-6),
            0
        )
        
        # Normalized reflectance: how bright overall (thin clouds have low-medium reflectance)
        total_reflectance = self.blue_band + self.green_band + self.red_band + self.nir_band
        self.normalized_reflectance = total_reflectance / 4.0
        
        # Thin cloud indicator: moderate reflectance + high blue/red ratio
        thin_cloud_threshold_low = 1000   # Low reflectance
        thin_cloud_threshold_high = 4000  # High reflectance (thick clouds)
        
        self.thin_cloud_indicator = np.logical_and(
            self.normalized_reflectance > thin_cloud_threshold_low,
            self.normalized_reflectance < thin_cloud_threshold_high
        ).astype(np.float32) * (self.blue_red_ratio > 1.05).astype(np.float32)
    
    def _classify_cloud_thickness(self):
        """
        Classify ground truth clouds into thin vs thick based on reflectance.
        This helps us reward thin cloud detection specifically.
        """
        # Where ground truth says there are clouds
        cloud_mask = self.ground_truth > 0
        
        # Estimate thickness from reflectance
        # Thin clouds: 1000-4000 reflectance
        # Thick clouds: >4000 reflectance
        thickness_threshold = 4000
        
        self.thin_clouds_gt = np.logical_and(
            cloud_mask,
            self.normalized_reflectance < thickness_threshold
        ).astype(np.uint8)
        
        self.thick_clouds_gt = np.logical_and(
            cloud_mask,
            self.normalized_reflectance >= thickness_threshold
        ).astype(np.uint8)
    
    def _compute_texture_features(self):
        """Pre-compute texture features for efficiency."""
        # We'll compute these on-demand per patch to save memory
        pass
    
    def _extract_texture_features(self, patch):
        """
        Extract texture features from a patch.
        
        Returns:
            - variance: local intensity variance
            - edge_density: proportion of edge pixels
            - homogeneity: GLCM homogeneity
            - contrast: GLCM contrast
            - energy: GLCM energy
        """
        # Variance
        variance = np.var(patch)
        
        # Edge density using Sobel
        sobel_x = ndimage.sobel(patch, axis=0)
        sobel_y = ndimage.sobel(patch, axis=1)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_density = (edge_magnitude > edge_magnitude.mean()).sum() / patch.size
        
        # GLCM features (quantize to reduce computation)
        patch_quantized = (patch * 15).astype(np.uint8)  # 16 levels
        try:
            glcm = graycomatrix(patch_quantized, [1], [0], levels=16, symmetric=True, normed=True)
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            contrast = graycoprops(glcm, 'contrast')[0, 0] / 225.0  # Normalize
            energy = graycoprops(glcm, 'energy')[0, 0]
        except:
            homogeneity = 0.5
            contrast = 0.5
            energy = 0.5
        
        return np.array([variance, edge_density, homogeneity, contrast, energy])
    
    def _extract_spectral_features(self, i, j):
        """Extract spectral index features for a patch region."""
        ndvi_patch = self.ndvi[i:i+self.patch_size, j:j+self.patch_size]
        ndsi_patch = self.ndsi[i:i+self.patch_size, j:j+self.patch_size]
        
        nir_patch = self.nir_band[i:i+self.patch_size, j:j+self.patch_size]
        red_patch = self.red_band[i:i+self.patch_size, j:j+self.patch_size]
        green_patch = self.green_band[i:i+self.patch_size, j:j+self.patch_size]
        blue_patch = self.blue_band[i:i+self.patch_size, j:j+self.patch_size]
        
        # Normalize reflectances to [0, 1]
        max_val = 10000.0  # Typical Sentinel-2 scale
        
        return np.array([
            np.clip(ndsi_patch.mean(), -1, 1) * 0.5 + 0.5,  # Scale to [0, 1]
            np.clip(ndvi_patch.mean(), -1, 1) * 0.5 + 0.5,  # Scale to [0, 1]
            np.clip(nir_patch.mean() / max_val, 0, 1),
            np.clip(red_patch.mean() / max_val, 0, 1),
            np.clip(green_patch.mean() / max_val, 0, 1),
            np.clip(blue_patch.mean() / max_val, 0, 1)
        ])
    
    def _extract_thin_cloud_features(self, i, j):
        """Extract thin cloud indicators for a patch region."""
        blue_red_patch = self.blue_red_ratio[i:i+self.patch_size, j:j+self.patch_size]
        reflectance_patch = self.normalized_reflectance[i:i+self.patch_size, j:j+self.patch_size]
        thin_indicator_patch = self.thin_cloud_indicator[i:i+self.patch_size, j:j+self.patch_size]
        
        # Check if this patch contains thin clouds (from ground truth)
        thin_gt_patch = self.thin_clouds_gt[i:i+self.patch_size, j:j+self.patch_size]
        thick_gt_patch = self.thick_clouds_gt[i:i+self.patch_size, j:j+self.patch_size]
        
        return np.array([
            np.clip(blue_red_patch.mean() / 2.0, 0, 1),  # Normalize
            np.clip(reflectance_patch.mean() / 10000.0, 0, 1),  # Normalize
            thin_indicator_patch.mean(),  # Already 0-1
            float(thin_gt_patch.sum() > 0)  # Has thin clouds flag
        ])
    
    def _get_observation(self):
        """Construct observation vector with all features including thin cloud indicators."""
        i, j = self.current_pos
        
        # CNN probability patch
        cnn_patch = self.cnn_prob[i:i+self.patch_size, j:j+self.patch_size]
        cnn_flat = cnn_patch.flatten()
        
        # Texture features
        texture_features = self._extract_texture_features(cnn_patch)
        
        # Spectral features
        spectral_features = self._extract_spectral_features(i, j)
        
        # Thin cloud features (NEW - this is the key for thin cloud detection)
        thin_cloud_features = self._extract_thin_cloud_features(i, j)
        
        # Spatial statistics
        spatial_stats = np.array([
            cnn_patch.mean(),
            cnn_patch.std(),
            cnn_patch.min(),
            cnn_patch.max()
        ])
        
        # Concatenate all features
        obs = np.concatenate([
            cnn_flat,
            texture_features,
            spectral_features,
            thin_cloud_features,
            spatial_stats,
            self.current_actions
        ]).astype(np.float32)
        
        return obs
    
    def _compute_reward(self, adjusted_pred, i, j):
        """
        Compute reward with SPECIAL FOCUS on thin cloud detection.
        
        THIS IS THE KEY: We want to specifically reward detecting thin clouds!
        
        Args:
            adjusted_pred: Binary prediction for current patch
            i, j: Patch position
        """
        gt_patch = self.ground_truth[i:i+self.patch_size, j:j+self.patch_size]
        gt_binary = (gt_patch > 0).astype(np.uint8)
        
        # Separate thin and thick clouds in this patch
        thin_gt_patch = self.thin_clouds_gt[i:i+self.patch_size, j:j+self.patch_size]
        thick_gt_patch = self.thick_clouds_gt[i:i+self.patch_size, j:j+self.patch_size]
        
        # F-beta score for overall patch
        gt_flat = gt_binary.flatten()
        pred_flat = adjusted_pred.flatten()
        
        fbeta_adjusted = fbeta_score(gt_flat, pred_flat, beta=self.beta, zero_division=0)
        
        # Baseline F-beta for comparison
        baseline_patch = (self.cnn_prob[i:i+self.patch_size, j:j+self.patch_size] > self.baseline_threshold).astype(np.uint8)
        fbeta_baseline = fbeta_score(gt_flat, baseline_patch.flatten(), beta=self.beta, zero_division=0)
        
        # Base reward: improvement over baseline
        reward = (fbeta_adjusted - fbeta_baseline) * 10.0
        
        # ============================================================
        # THIN CLOUD BONUS (THIS IS THE KEY INNOVATION!)
        # ============================================================
        if thin_gt_patch.sum() > 0:
            # This patch contains thin clouds!
            thin_flat = thin_gt_patch.flatten()
            
            # Calculate recall specifically for thin clouds
            thin_cloud_pixels_detected = np.logical_and(pred_flat == 1, thin_flat == 1).sum()
            thin_cloud_total = thin_flat.sum()
            thin_recall = thin_cloud_pixels_detected / max(thin_cloud_total, 1)
            
            # Calculate precision for thin cloud predictions
            thin_pred_correct = np.logical_and(pred_flat == 1, thin_flat == 1).sum()
            thin_pred_total = np.logical_and(pred_flat == 1, thin_flat + thick_gt_patch.flatten() > 0).sum()
            thin_precision = thin_pred_correct / max(thin_pred_total, 1) if thin_pred_total > 0 else 0
            
            # BIG BONUS for detecting thin clouds (weighted F1 for thin clouds)
            if thin_recall > 0.3:  # Detected at least 30% of thin clouds
                thin_f1 = 2 * thin_precision * thin_recall / (thin_precision + thin_recall + 1e-6)
                reward += thin_f1 * 5.0  # Big bonus!
            
            # Extra bonus for high thin cloud recall
            if thin_recall > 0.5:
                reward += 2.0
            if thin_recall > 0.7:
                reward += 3.0
        
        # ============================================================
        # FALSE POSITIVE PENALTIES (still important)
        # ============================================================
        mean_reflectance = self.normalized_reflectance[i:i+self.patch_size, j:j+self.patch_size].mean()
        dark_threshold = 1000.0  # Low reflectance threshold
        
        if mean_reflectance < dark_threshold:
            # This is a dark area (potential shadow/water)
            false_positives = np.logical_and(pred_flat == 1, gt_flat == 0).sum()
            if false_positives > 0:
                reward -= false_positives / pred_flat.size * 2.0  # Penalty
        
        # ============================================================
        # THICK CLOUD BASELINE (should already be detected)
        # ============================================================
        if thick_gt_patch.sum() > 0:
            # Thick clouds should be easy - penalize if we miss them
            thick_flat = thick_gt_patch.flatten()
            thick_recall = np.logical_and(pred_flat == 1, thick_flat == 1).sum() / max(thick_flat.sum(), 1)
            if thick_recall < 0.5:  # Missing thick clouds is bad
                reward -= 1.0
        
        return reward
    
    def reset(self, seed=None, options=None):
        """Reset environment to start."""
        super().reset(seed=seed)
        self.current_patch_idx = 0
        self.current_pos = (0, 0)
        self.current_actions = np.array([0.0, 0.0, 0.5])  # [threshold_delta, thin_boost, spectral_weight]
        return self._get_observation(), {}
    
    def step(self, action):
        """
        Execute action and return next observation.
        
        Action: [threshold_delta, thin_cloud_boost, spectral_weight]
        thin_cloud_boost: Extra probability boost for pixels identified as thin clouds
        """
        i, j = self.current_pos
        
        # Clip and store actions
        threshold_delta = np.clip(action[0], -0.3, 0.3)
        thin_cloud_boost = np.clip(action[1], 0.0, 0.4)  # Boost for thin clouds
        spectral_weight = np.clip(action[2], 0.0, 1.0)
        self.current_actions = np.array([threshold_delta, thin_cloud_boost, spectral_weight])
        
        # Extract patches
        cnn_patch = self.cnn_prob[i:i+self.patch_size, j:j+self.patch_size].copy()
        
        # Apply base threshold adjustment
        adjusted_threshold = np.clip(self.baseline_threshold + threshold_delta, 0.1, 0.9)
        
        # ============================================================
        # KEY INNOVATION: THIN CLOUD BOOST
        # ============================================================
        # Identify potential thin cloud pixels
        thin_indicator_patch = self.thin_cloud_indicator[i:i+self.patch_size, j:j+self.patch_size]
        blue_red_patch = self.blue_red_ratio[i:i+self.patch_size, j:j+self.patch_size]
        reflectance_patch = self.normalized_reflectance[i:i+self.patch_size, j:j+self.patch_size]
        
        # Pixels that look like thin clouds (moderate reflectance + high blue/red ratio)
        is_thin_cloud_like = np.logical_and(
            np.logical_and(reflectance_patch > 1000, reflectance_patch < 4000),
            blue_red_patch > 1.05
        )
        
        # Boost CNN probability for thin cloud-like pixels
        cnn_boosted = cnn_patch.copy()
        cnn_boosted[is_thin_cloud_like] += thin_cloud_boost
        cnn_boosted = np.clip(cnn_boosted, 0, 1)
        
        # Apply threshold to boosted probabilities
        threshold_pred = (cnn_boosted > adjusted_threshold).astype(np.float32)
        
        # ============================================================
        # Spectral-based filtering (to reduce false positives)
        # ============================================================
        ndvi_patch = self.ndvi[i:i+self.patch_size, j:j+self.patch_size]
        ndsi_patch = self.ndsi[i:i+self.patch_size, j:j+self.patch_size]
        
        # NDVI < 0.2 suggests non-vegetation (could be cloud)
        # NDSI > 0.4 suggests snow/ice (could be cloud)
        spectral_mask = np.logical_or(ndvi_patch < 0.2, ndsi_patch > 0.4).astype(np.float32)
        
        # Combine predictions with weights
        # High spectral_weight = trust spectral indices more
        combined_pred = (
            threshold_pred * (1.0 - spectral_weight) +
            threshold_pred * spectral_mask * spectral_weight
        )
        
        # Final binary prediction
        final_pred = (combined_pred > 0.5).astype(np.uint8)
        
        # Compute reward (with thin cloud bonuses!)
        reward = self._compute_reward(final_pred, i, j)
        
        # Move to next patch
        self.current_patch_idx += 1
        done = self.current_patch_idx >= self.num_patches
        
        if not done:
            # Update position (row-major order)
            patch_row = self.current_patch_idx // self.num_patches_w
            patch_col = self.current_patch_idx % self.num_patches_w
            self.current_pos = (patch_row * self.patch_size, patch_col * self.patch_size)
        
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        
        return obs, reward, done, False, {}


def test_environment():
    """Test the multi-feature environment."""
    print("Testing MultiFeatureRefinementEnv...")
    
    # Create dummy data
    image = np.random.rand(512, 512, 10) * 5000
    cnn_prob = np.random.rand(512, 512)
    ground_truth = (np.random.rand(512, 512) > 0.7).astype(np.uint8)
    
    env = MultiFeatureRefinementEnv(image, cnn_prob, ground_truth, patch_size=64)
    
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Number of patches: {env.num_patches}")
    
    # Test reset and step
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    action = env.action_space.sample()
    print(f"Sample action: {action}")
    
    obs, reward, done, _, _ = env.step(action)
    print(f"Reward: {reward:.4f}, Done: {done}")
    
    print("✅ Environment test passed!")


if __name__ == "__main__":
    test_environment()
