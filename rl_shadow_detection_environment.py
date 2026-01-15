"""
Phase 2: Shadow Detection RL Environment

GOAL: Reduce false positives by identifying and filtering shadow pixels

Starting from RL Threshold baseline (30.25% F1, 20% precision)
Problem: Many false positives are shadows, dark terrain, water bodies
Solution: Learn to distinguish shadows from clouds using spectral + spatial features

Key Insight: Shadows and clouds both appear dark but have different properties:
- Shadows: Very low reflectance, vegetation NDVI visible, sharp edges near objects
- Clouds: Variable reflectance, low NDVI (no vegetation), smoother boundaries

Author: Thesis Implementation - Phase 2
Date: January 2026
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
from scipy import ndimage


class ShadowDetectionEnv(gym.Env):
    """
    RL environment for shadow detection and filtering.
    
    Builds on RL Threshold baseline (30.25% F1) by adding shadow filtering.
    
    Observation Space:
        - CNN probability patch (64x64)
        - Spectral features (NDVI, NIR/Red ratio, brightness)
        - Texture features (edge strength, variance)
        - Spatial context (gradient magnitude, neighbor statistics)
        
    Action Space:
        - threshold_delta: [-0.3, +0.3] - Base threshold adjustment (from Phase 1)
        - shadow_filter_strength: [0, 1.0] - How aggressively to filter shadows
        
    Reward:
        - Base: F-beta score (beta=0.7, emphasizes precision)
        - BONUS: Extra reward for precision improvement (reducing false positives)
        - PENALTY: For missing true clouds
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
        self.beta = beta
        
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
        
        # Pre-compute shadow indicators
        self._compute_shadow_indicators()
        
        # Action space: [threshold_delta, shadow_filter_strength]
        # CONSTRAINED to prevent extremes that led to "predict nothing" policy
        self.action_space = spaces.Box(
            low=np.array([-0.15, 0.0]),   # Reduced from -0.3 to -0.15
            high=np.array([0.15, 0.6]),   # Reduced from 0.3/1.0 to 0.15/0.6
            dtype=np.float32
        )
        
        # Observation space
        patch_features = patch_size * patch_size  # CNN prob
        shadow_features = 6  # NDVI, NIR/Red, brightness, edge_strength, variance, gradient_mag
        spatial_stats = 4  # mean, std, min, max
        action_memory = 2  # previous actions
        
        obs_size = patch_features + shadow_features + spatial_stats + action_memory
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        
        # State
        self.current_patch_idx = 0
        self.current_pos = (0, 0)
        self.current_actions = np.array([0.0, 0.5])  # [threshold_delta, shadow_filter_strength]
        
    def _compute_shadow_indicators(self):
        """
        Compute features that distinguish shadows from clouds.
        
        Shadow characteristics:
        - Very low overall brightness (darker than thin clouds)
        - Vegetation NDVI still visible (plants underneath shadow)
        - Sharp edges adjacent to structures
        - Low blue scattering (unlike clouds)
        """
        if self.image.shape[2] >= 8:
            blue = self.image[:, :, 1].astype(np.float32)
            green = self.image[:, :, 2].astype(np.float32)
            red = self.image[:, :, 3].astype(np.float32)
            nir = self.image[:, :, 7].astype(np.float32)
            
            # NDVI: Shadows over vegetation show positive NDVI
            denominator = nir + red
            self.ndvi = np.where(denominator != 0, (nir - red) / denominator, 0)
            
            # NIR/Red ratio: Different for shadows vs clouds
            self.nir_red_ratio = nir / (red + 1e-6)
            
            # Overall brightness: Shadows are very dark
            self.brightness = (blue + green + red + nir) / 4.0
            
            # Blue band alone: Clouds scatter blue, shadows don't
            self.blue_band = blue
            
            # Compute edge strength (shadows have sharp edges)
            self.edge_strength = self._compute_edge_strength()
            
            # Compute local variance (shadows more uniform than clouds)
            self.local_variance = self._compute_local_variance()
            
        else:
            # Fallback
            self.ndvi = np.zeros_like(self.cnn_prob)
            self.nir_red_ratio = np.zeros_like(self.cnn_prob)
            self.brightness = np.zeros_like(self.cnn_prob)
            self.blue_band = np.zeros_like(self.cnn_prob)
            self.edge_strength = np.zeros_like(self.cnn_prob)
            self.local_variance = np.zeros_like(self.cnn_prob)
    
    def _compute_edge_strength(self):
        """Compute gradient magnitude (shadows have sharp edges)."""
        brightness = self.brightness
        
        # Sobel edge detection
        sobel_x = ndimage.sobel(brightness, axis=1)
        sobel_y = ndimage.sobel(brightness, axis=0)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize to 0-1
        if gradient_magnitude.max() > 0:
            gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
        
        return gradient_magnitude
    
    def _compute_local_variance(self):
        """Compute local variance in sliding window."""
        from scipy.ndimage import uniform_filter
        
        # Use 5x5 window
        mean = uniform_filter(self.brightness, size=5)
        mean_sq = uniform_filter(self.brightness**2, size=5)
        variance = mean_sq - mean**2
        
        # Normalize
        if variance.max() > 0:
            variance = variance / variance.max()
        
        return variance
    
    def _compute_shadow_likelihood(self, patch_data):
        """
        Estimate shadow likelihood for each pixel in patch.
        
        High shadow likelihood if:
        - Very low brightness
        - Positive NDVI (vegetation visible)
        - Sharp edges nearby
        - Low blue scattering
        
        Returns: shadow_score [0, 1] for each pixel
        """
        ndvi = patch_data['ndvi']
        brightness = patch_data['brightness']
        edge_strength = patch_data['edge_strength']
        blue = patch_data['blue']
        
        # Normalize all to 0-1 range for scoring
        # Very low brightness suggests shadow
        darkness_score = 1.0 - brightness  # Higher = darker
        
        # Positive NDVI suggests vegetation (visible in shadows, not in clouds)
        vegetation_score = np.clip(ndvi, 0, 1)
        
        # High edge strength suggests shadow boundary
        edge_score = edge_strength
        
        # Low blue scattering suggests shadow (clouds scatter blue)
        low_blue_score = 1.0 - blue
        
        # Combine scores (weighted average)
        shadow_score = (
            0.4 * darkness_score +
            0.3 * vegetation_score +
            0.2 * edge_score +
            0.1 * low_blue_score
        )
        
        return np.clip(shadow_score, 0, 1)
    
    def _get_observation(self):
        """Get observation for current patch."""
        i, j = self.current_pos
        
        # CNN probability patch (flattened)
        cnn_patch = self.cnn_prob[i:i+self.patch_size, j:j+self.patch_size]
        cnn_flat = cnn_patch.flatten()
        
        # Shadow indicator features (patch statistics)
        ndvi_patch = self.ndvi[i:i+self.patch_size, j:j+self.patch_size]
        nir_red_patch = self.nir_red_ratio[i:i+self.patch_size, j:j+self.patch_size]
        brightness_patch = self.brightness[i:i+self.patch_size, j:j+self.patch_size]
        edge_patch = self.edge_strength[i:i+self.patch_size, j:j+self.patch_size]
        variance_patch = self.local_variance[i:i+self.patch_size, j:j+self.patch_size]
        blue_patch = self.blue_band[i:i+self.patch_size, j:j+self.patch_size]
        
        shadow_features = np.array([
            ndvi_patch.mean(),
            nir_red_patch.mean(),
            brightness_patch.mean(),
            edge_patch.mean(),
            variance_patch.mean(),
            blue_patch.mean()
        ])
        
        # Spatial statistics
        spatial_stats = np.array([
            cnn_patch.mean(),
            cnn_patch.std(),
            cnn_patch.min(),
            cnn_patch.max()
        ])
        
        # Previous actions
        action_memory = self.current_actions
        
        # Concatenate
        obs = np.concatenate([
            cnn_flat,
            shadow_features,
            spatial_stats,
            action_memory
        ])
        
        return obs.astype(np.float32)
    
    def _compute_reward(self, prediction, i, j, threshold_delta, shadow_filter_strength):
        """
        Compute reward for the prediction.
        
        REDESIGNED to prevent "predict nothing" degenerate policy:
        1. Strong penalties for catastrophic behaviors
        2. Balanced precision/recall requirements
        3. Action regularization to discourage extremes
        4. F-beta base reward without easy-to-hack bonuses
        """
        gt_patch = self.ground_truth[i:i+self.patch_size, j:j+self.patch_size]
        gt_binary = (gt_patch > 0).astype(np.uint8).flatten()
        pred_flat = prediction.flatten()
        
        # CATASTROPHIC PENALTIES - prevent reward hacking
        if pred_flat.sum() == 0:  # Predicting nothing
            return -50.0
        if pred_flat.sum() == len(pred_flat):  # Predicting everything
            return -50.0
        
        # Base metrics
        fbeta = fbeta_score(gt_binary, pred_flat, beta=self.beta, zero_division=0)
        precision = precision_score(gt_binary, pred_flat, zero_division=0)
        recall = recall_score(gt_binary, pred_flat, zero_division=0)
        f1 = f1_score(gt_binary, pred_flat, zero_division=0)
        
        # Base reward: F-beta score scaled to reasonable range
        reward = fbeta * 10.0  # 0-10 range
        
        # HARD FLOOR on recall - prevent excessive filtering
        if recall < 0.15:
            reward -= 20.0  # Massive penalty
        elif recall < 0.25:
            reward -= 10.0
        
        # HARD FLOOR on precision - prevent spam predictions
        if precision < 0.10:
            reward -= 15.0
        elif precision < 0.15:
            reward -= 5.0
        
        # Bonus for balanced improvement (both precision AND recall matter)
        if precision > 0.25 and recall > 0.30:
            reward += 3.0
        if f1 > 0.30:  # Beat Phase 1 ceiling
            reward += 5.0
        
        # ACTION REGULARIZATION - penalize extreme actions
        # Encourages moderate adjustments over drastic changes
        action_penalty = 0.5 * (abs(threshold_delta / 0.15) + abs(shadow_filter_strength / 0.6))
        reward -= action_penalty
        
        return reward
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        self.current_patch_idx = 0
        self.current_pos = (0, 0)
        self.current_actions = np.array([0.0, 0.5])
        return self._get_observation(), {}
    
    def step(self, action):
        """Take a step in the environment."""
        i, j = self.current_pos
        
        # Clip and store actions (with new constrained ranges)
        threshold_delta = np.clip(action[0], -0.15, 0.15)
        shadow_filter_strength = np.clip(action[1], 0.0, 0.6)
        self.current_actions = np.array([threshold_delta, shadow_filter_strength])
        
        # Extract patches
        cnn_patch = self.cnn_prob[i:i+self.patch_size, j:j+self.patch_size].copy()
        
        # Apply base threshold adjustment
        adjusted_threshold = np.clip(self.baseline_threshold + threshold_delta, 0.1, 0.9)
        
        # Get initial prediction
        initial_pred = (cnn_patch > adjusted_threshold).astype(np.float32)
        
        # ============================================================
        # SHADOW FILTERING
        # ============================================================
        # Compute shadow likelihood for each pixel
        patch_data = {
            'ndvi': self.ndvi[i:i+self.patch_size, j:j+self.patch_size],
            'brightness': self.brightness[i:i+self.patch_size, j:j+self.patch_size],
            'edge_strength': self.edge_strength[i:i+self.patch_size, j:j+self.patch_size],
            'blue': self.blue_band[i:i+self.patch_size, j:j+self.patch_size]
        }
        
        shadow_score = self._compute_shadow_likelihood(patch_data)
        
        # Filter predictions based on shadow likelihood
        # If shadow_filter_strength is high and shadow_score is high, remove prediction
        shadow_mask = (shadow_score < (1.0 - shadow_filter_strength)).astype(np.float32)
        
        # Apply shadow filter
        final_pred = (initial_pred * shadow_mask).astype(np.uint8)
        
        # Compute reward (pass actions for regularization)
        reward = self._compute_reward(final_pred, i, j, threshold_delta, shadow_filter_strength)
        
        # Move to next patch
        self.current_patch_idx += 1
        done = self.current_patch_idx >= self.num_patches
        
        if not done:
            patch_row = self.current_patch_idx // self.num_patches_w
            patch_col = self.current_patch_idx % self.num_patches_w
            self.current_pos = (patch_row * self.patch_size, patch_col * self.patch_size)
        
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        
        return obs, reward, done, False, {}


def test_environment():
    """Test the shadow detection environment."""
    print("Testing ShadowDetectionEnv...")
    
    # Create dummy data
    image = np.random.rand(512, 512, 10)
    cnn_prob = np.random.rand(512, 512)
    ground_truth = (np.random.rand(512, 512) > 0.7).astype(np.uint8)
    
    env = ShadowDetectionEnv(image, cnn_prob, ground_truth, patch_size=64)
    
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Number of patches: {env.num_patches}")
    
    # Test reset and step
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Random action
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    print(f"After step - Obs shape: {obs.shape}, Reward: {reward:.2f}, Done: {done}")
    
    print("✅ Environment test passed!")


if __name__ == "__main__":
    test_environment()
