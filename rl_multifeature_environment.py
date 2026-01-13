"""
Multi-Feature RL Environment for Cloud Detection Refinement

Enhanced environment that uses:
- Texture features (variance, edge density, homogeneity)
- Spectral indices (NDSI, NDVI) 
- Multi-dimensional actions (threshold, texture weight, spectral mask)
- Modified reward with F-beta and shadow penalties

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
    Enhanced RL environment for cloud detection with multiple features and actions.
    
    Observation Space:
        - CNN probability patch (64x64 flattened)
        - Texture features (variance, edge density, homogeneity)
        - Spectral indices (NDSI, NDVI, mean reflectances)
        - Spatial statistics (mean, std, min, max)
        - Current action values
        
    Action Space:
        - threshold_delta: [-0.3, +0.3] - Adjust CNN threshold
        - texture_weight: [0, 1] - Weight for texture-based masking
        - spectral_weight: [0, 1] - Weight for spectral index masking
        
    Reward:
        - F-beta score (beta=0.7) to emphasize precision
        - Penalties for false positives in dark areas
        - Bonuses for correctly handling shadows
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
        
        # Pre-compute texture features for all patches
        self._compute_texture_features()
        
        # Action space: [threshold_delta, texture_weight, spectral_weight]
        self.action_space = spaces.Box(
            low=np.array([-0.3, 0.0, 0.0]),
            high=np.array([0.3, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space calculation
        patch_features = patch_size * patch_size  # CNN prob patch
        texture_features = 5  # variance, edge_density, homogeneity, contrast, energy
        spectral_features = 6  # NDSI, NDVI, mean_nir, mean_red, mean_green, mean_swir
        spatial_stats = 4  # mean, std, min, max
        action_memory = 3  # previous actions
        
        obs_size = patch_features + texture_features + spectral_features + spatial_stats + action_memory
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        
        # State
        self.current_patch_idx = 0
        self.current_pos = (0, 0)
        self.current_actions = np.array([0.0, 0.5, 0.5])  # Initial values
        
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
    
    def _get_observation(self):
        """Construct observation vector with all features."""
        i, j = self.current_pos
        
        # CNN probability patch
        cnn_patch = self.cnn_prob[i:i+self.patch_size, j:j+self.patch_size]
        cnn_flat = cnn_patch.flatten()
        
        # Texture features
        texture_features = self._extract_texture_features(cnn_patch)
        
        # Spectral features
        spectral_features = self._extract_spectral_features(i, j)
        
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
            spatial_stats,
            self.current_actions
        ]).astype(np.float32)
        
        return obs
    
    def _compute_reward(self, adjusted_pred, i, j):
        """
        Compute reward with F-beta score and shadow penalties.
        
        Args:
            adjusted_pred: Binary prediction for current patch
            i, j: Patch position
        """
        gt_patch = self.ground_truth[i:i+self.patch_size, j:j+self.patch_size]
        gt_binary = (gt_patch > 0).astype(np.uint8)
        
        # F-beta score for this patch
        gt_flat = gt_binary.flatten()
        pred_flat = adjusted_pred.flatten()
        
        fbeta_adjusted = fbeta_score(gt_flat, pred_flat, beta=self.beta, zero_division=0)
        
        # Baseline F-beta for comparison
        baseline_patch = (self.cnn_prob[i:i+self.patch_size, j:j+self.patch_size] > self.baseline_threshold).astype(np.uint8)
        fbeta_baseline = fbeta_score(gt_flat, baseline_patch.flatten(), beta=self.beta, zero_division=0)
        
        # Base reward: improvement over baseline
        reward = (fbeta_adjusted - fbeta_baseline) * 10.0
        
        # Penalty for false positives in dark areas
        mean_reflectance = self.nir_band[i:i+self.patch_size, j:j+self.patch_size].mean()
        dark_threshold = 1000.0  # Low reflectance threshold
        
        if mean_reflectance < dark_threshold:
            # This is a dark area (potential shadow/water)
            false_positives = np.logical_and(pred_flat == 1, gt_flat == 0).sum()
            if false_positives > 0:
                reward -= false_positives / pred_flat.size * 2.0  # Penalty
        
        # Bonus for high precision in dark areas
        if mean_reflectance < dark_threshold:
            precision = precision_score(gt_flat, pred_flat, zero_division=0)
            if precision > 0.5:
                reward += precision * 1.0
        
        # Bonus for correctly rejecting dark areas as non-cloud
        if mean_reflectance < dark_threshold and gt_binary.sum() == 0:
            if pred_flat.sum() < gt_flat.size * 0.1:  # Correctly sparse prediction
                reward += 0.5
        
        return reward
    
    def reset(self, seed=None, options=None):
        """Reset environment to start."""
        super().reset(seed=seed)
        self.current_patch_idx = 0
        self.current_pos = (0, 0)
        self.current_actions = np.array([0.0, 0.5, 0.5])
        return self._get_observation(), {}
    
    def step(self, action):
        """
        Execute action and return next observation.
        
        Action: [threshold_delta, texture_weight, spectral_weight]
        """
        i, j = self.current_pos
        
        # Clip and store actions
        threshold_delta = np.clip(action[0], -0.3, 0.3)
        texture_weight = np.clip(action[1], 0.0, 1.0)
        spectral_weight = np.clip(action[2], 0.0, 1.0)
        self.current_actions = np.array([threshold_delta, texture_weight, spectral_weight])
        
        # Extract patches
        cnn_patch = self.cnn_prob[i:i+self.patch_size, j:j+self.patch_size]
        
        # Apply threshold adjustment
        adjusted_threshold = np.clip(self.baseline_threshold + threshold_delta, 0.1, 0.9)
        threshold_pred = (cnn_patch > adjusted_threshold).astype(np.float32)
        
        # Texture-based masking
        texture_features = self._extract_texture_features(cnn_patch)
        texture_variance = texture_features[0]
        # Clouds typically have more texture variation
        texture_mask = (texture_variance > 0.01).astype(np.float32)
        
        # Spectral-based masking
        ndvi_patch = self.ndvi[i:i+self.patch_size, j:j+self.patch_size]
        ndsi_patch = self.ndsi[i:i+self.patch_size, j:j+self.patch_size]
        
        # NDVI < 0.2 suggests non-vegetation (could be cloud)
        # NDSI > 0.4 suggests snow/ice (could be cloud)
        spectral_mask = np.logical_or(ndvi_patch < 0.2, ndsi_patch > 0.4).astype(np.float32)
        
        # Combine predictions with weights
        combined_pred = (
            threshold_pred * (1.0 - texture_weight - spectral_weight) +
            threshold_pred * texture_mask * texture_weight +
            threshold_pred * spectral_mask * spectral_weight
        )
        
        # Final binary prediction
        final_pred = (combined_pred > 0.5).astype(np.uint8)
        
        # Compute reward
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
