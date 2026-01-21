"""
Discrete Action Environment for Thin Cloud Detection (DQN Compatible)
Converts the continuous action space to discrete for DQN comparison.

Action Space: 15 discrete actions
- 5 threshold_delta options: [-0.2, -0.1, 0.0, 0.1, 0.2]
- 3 thin_boost options: [0.0, 0.15, 0.3]
- Combined: 5 × 3 = 15 actions
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ThinCloudDetectionEnvDiscrete(gym.Env):
    """
    Discrete action version for DQN comparison.
    Same observation space as continuous version, but with discrete actions.
    """
    
    # Discrete action mappings
    THRESHOLD_OPTIONS = [-0.2, -0.1, 0.0, 0.1, 0.2]  # 5 options
    BOOST_OPTIONS = [0.0, 0.15, 0.3]  # 3 options
    
    def __init__(self, image, cnn_prob, ground_truth, patch_size=64):
        super().__init__()
        
        self.image = image
        self.cnn_prob = cnn_prob
        self.ground_truth = ground_truth
        self.patch_size = patch_size
        
        self.height, self.width = ground_truth.shape
        
        # Calculate number of patches
        self.n_patches_h = self.height // patch_size
        self.n_patches_w = self.width // patch_size
        self.num_patches = self.n_patches_h * self.n_patches_w
        
        # 15 discrete actions (5 threshold × 3 boost)
        self.action_space = spaces.Discrete(len(self.THRESHOLD_OPTIONS) * len(self.BOOST_OPTIONS))
        
        # Same observation space as continuous version (20 features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )
        
        # Precompute thin cloud indicator
        self._compute_thin_cloud_indicator()
        
        # Initialize
        self.current_patch = 0
        self.current_pos = (0, 0)
        self.refined_mask = np.zeros_like(ground_truth, dtype=np.uint8)
        self.cumulative_reward = 0
        
    def _compute_thin_cloud_indicator(self):
        """Compute thin cloud indicator using spectral features."""
        # Blue band (B2) and Red band (B4)
        blue = self.image[:, :, 0] if self.image.shape[2] > 0 else np.zeros_like(self.cnn_prob)
        red = self.image[:, :, 2] if self.image.shape[2] > 2 else np.zeros_like(self.cnn_prob)
        
        # Normalize
        blue_norm = blue / (np.max(blue) + 1e-8)
        red_norm = red / (np.max(red) + 1e-8)
        
        # Thin cloud indicators:
        # 1. High blue/red ratio (thin clouds scatter blue more)
        # 2. Moderate CNN probability (not confident thick clouds)
        # 3. Low overall reflectance
        
        blue_red_ratio = blue_norm / (red_norm + 1e-8)
        moderate_prob = (self.cnn_prob > 0.2) & (self.cnn_prob < 0.7)
        high_ratio = blue_red_ratio > np.percentile(blue_red_ratio, 70)
        
        # Combine indicators
        self.thin_cloud_indicator = (
            moderate_prob.astype(float) * 0.5 +
            high_ratio.astype(float) * 0.3 +
            (1 - self.cnn_prob) * 0.2  # Lower CNN confidence suggests thin
        )
        
    def _decode_action(self, action):
        """Convert discrete action index to threshold_delta and thin_boost."""
        n_boost = len(self.BOOST_OPTIONS)
        threshold_idx = action // n_boost
        boost_idx = action % n_boost
        
        threshold_delta = self.THRESHOLD_OPTIONS[threshold_idx]
        thin_boost = self.BOOST_OPTIONS[boost_idx]
        
        return threshold_delta, thin_boost
        
    def _get_patch_features(self):
        """Extract 20 features for current patch."""
        i, j = self.current_pos
        ps = self.patch_size
        
        # Get patch data
        cnn_patch = self.cnn_prob[i:i+ps, j:j+ps]
        gt_patch = self.ground_truth[i:i+ps, j:j+ps]
        thin_patch = self.thin_cloud_indicator[i:i+ps, j:j+ps]
        image_patch = self.image[i:i+ps, j:j+ps, :]
        
        # CNN probability features (5)
        cnn_mean = np.mean(cnn_patch)
        cnn_std = np.std(cnn_patch)
        cnn_max = np.max(cnn_patch)
        cnn_min = np.min(cnn_patch)
        cnn_median = np.median(cnn_patch)
        
        # Thin cloud indicator features (5)
        thin_mean = np.mean(thin_patch)
        thin_std = np.std(thin_patch)
        thin_max = np.max(thin_patch)
        thin_coverage = np.mean(thin_patch > 0.5)
        thin_cnn_corr = np.corrcoef(thin_patch.flatten(), cnn_patch.flatten())[0, 1]
        if np.isnan(thin_cnn_corr):
            thin_cnn_corr = 0.0
        
        # Spectral features (6)
        blue = image_patch[:, :, 0] if image_patch.shape[2] > 0 else np.zeros((ps, ps))
        green = image_patch[:, :, 1] if image_patch.shape[2] > 1 else np.zeros((ps, ps))
        red = image_patch[:, :, 2] if image_patch.shape[2] > 2 else np.zeros((ps, ps))
        nir = image_patch[:, :, 3] if image_patch.shape[2] > 3 else np.zeros((ps, ps))
        
        blue_mean = np.mean(blue) / 10000.0  # Normalize
        green_mean = np.mean(green) / 10000.0
        red_mean = np.mean(red) / 10000.0
        nir_mean = np.mean(nir) / 10000.0
        brightness = (blue_mean + green_mean + red_mean) / 3
        blue_red_ratio = blue_mean / (red_mean + 1e-8)
        
        # Spatial features (4)
        edges = np.abs(np.diff(cnn_patch, axis=0)).mean() + np.abs(np.diff(cnn_patch, axis=1)).mean()
        homogeneity = 1.0 / (1.0 + cnn_std)
        cloud_fraction = np.mean(cnn_patch > 0.5)
        uncertain_fraction = np.mean((cnn_patch > 0.3) & (cnn_patch < 0.7))
        
        features = np.array([
            # CNN features (5)
            cnn_mean, cnn_std, cnn_max, cnn_min, cnn_median,
            # Thin cloud features (5)
            thin_mean, thin_std, thin_max, thin_coverage, thin_cnn_corr,
            # Spectral features (6)
            blue_mean, green_mean, red_mean, nir_mean, brightness, blue_red_ratio,
            # Spatial features (4)
            edges, homogeneity, cloud_fraction, uncertain_fraction
        ], dtype=np.float32)
        
        return features
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_patch = 0
        self.current_pos = (0, 0)
        self.refined_mask = np.zeros_like(self.ground_truth, dtype=np.uint8)
        self.cumulative_reward = 0
        
        return self._get_patch_features(), {}
        
    def step(self, action):
        """Execute discrete action."""
        # Decode discrete action
        threshold_delta, thin_boost = self._decode_action(action)
        
        i, j = self.current_pos
        ps = self.patch_size
        
        # Get patches
        cnn_patch = self.cnn_prob[i:i+ps, j:j+ps].copy()
        gt_patch = self.ground_truth[i:i+ps, j:j+ps]
        thin_indicator = self.thin_cloud_indicator[i:i+ps, j:j+ps]
        
        # Apply action: boost thin cloud regions and adjust threshold
        boosted_prob = np.clip(cnn_patch + thin_indicator * thin_boost, 0, 1)
        refined_patch = (boosted_prob > (0.5 + threshold_delta)).astype(np.uint8)
        
        # Store refined prediction
        self.refined_mask[i:i+ps, j:j+ps] = refined_patch
        
        # Calculate reward (multi-objective: thin cloud IoU + F1)
        reward = self._calculate_reward(refined_patch, gt_patch, thin_indicator)
        self.cumulative_reward += reward
        
        # Move to next patch
        self.current_patch += 1
        
        if self.current_patch >= self.num_patches:
            done = True
        else:
            done = False
            # Calculate next position
            patch_row = self.current_patch // self.n_patches_w
            patch_col = self.current_patch % self.n_patches_w
            self.current_pos = (patch_row * ps, patch_col * ps)
        
        truncated = False
        info = {
            'cumulative_reward': self.cumulative_reward,
            'threshold_delta': threshold_delta,
            'thin_boost': thin_boost
        }
        
        obs = self._get_patch_features() if not done else np.zeros(20, dtype=np.float32)
        
        return obs, reward, done, truncated, info
        
    def _calculate_reward(self, pred, gt, thin_indicator):
        """Multi-objective reward: 70% thin cloud IoU + 30% F1."""
        # Thin cloud mask (ground truth clouds with high thin indicator)
        thin_mask = (gt == 1) & (thin_indicator > 0.3)
        
        # Thin cloud IoU
        if thin_mask.sum() > 0:
            thin_intersection = np.sum((pred == 1) & thin_mask)
            thin_union = np.sum((pred == 1) | thin_mask)
            thin_iou = thin_intersection / (thin_union + 1e-8)
        else:
            thin_iou = 0.5  # Neutral if no thin clouds
        
        # Overall F1
        tp = np.sum((pred == 1) & (gt == 1))
        fp = np.sum((pred == 1) & (gt == 0))
        fn = np.sum((pred == 0) & (gt == 1))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Multi-objective reward
        reward = 0.7 * thin_iou + 0.3 * f1
        
        return reward


# Test the environment
if __name__ == "__main__":
    print("Testing ThinCloudDetectionEnvDiscrete...")
    
    # Create dummy data
    image = np.random.rand(256, 256, 13).astype(np.float32) * 10000
    cnn_prob = np.random.rand(256, 256).astype(np.float32)
    gt = (np.random.rand(256, 256) > 0.7).astype(np.uint8)
    
    env = ThinCloudDetectionEnvDiscrete(image, cnn_prob, gt, patch_size=64)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Number of patches: {env.num_patches}")
    
    # Test action decoding
    for action in [0, 7, 14]:
        td, tb = env._decode_action(action)
        print(f"Action {action} -> threshold_delta={td}, thin_boost={tb}")
    
    # Test episode
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    total_reward = 0
    steps = 0
    while True:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break
    
    print(f"Episode finished: {steps} steps, total reward: {total_reward:.4f}")
    print("✅ Discrete environment test passed!")
