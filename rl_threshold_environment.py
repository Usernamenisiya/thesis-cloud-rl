"""
RL Environment for Adaptive Threshold Refinement
Agent learns to adjust CNN thresholds per patch to maximize F1-score
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ThresholdRefinementEnv(gym.Env):
    """
    RL environment for learning adaptive thresholds.
    
    Observation Space:
        - CNN probability patch (64x64)
        - Spatial features (mean, std, local context)
        - Current threshold value
        
    Action Space:
        - Continuous threshold adjustment: delta in [-0.3, +0.3]
        
    Reward:
        - F1-score improvement over baseline threshold (0.5)
    """
    
    def __init__(self, image, cnn_prob, ground_truth, patch_size=64, baseline_threshold=0.5):
        super().__init__()
        
        # Handle both (H, W, 10) and (10, H, W) formats
        if image.ndim == 3:
            if image.shape[2] == 10:
                # Input is (H, W, 10), transpose to (10, H, W)
                self.image = np.transpose(image, (2, 0, 1))
            else:
                # Already (10, H, W)
                self.image = image
        else:
            raise ValueError(f"Expected 3D image, got shape {image.shape}")
        
        self.cnn_prob = cnn_prob  # (H, W)
        self.ground_truth = ground_truth  # (H, W)
        self.patch_size = patch_size
        self.baseline_threshold = baseline_threshold
        
        # Validate shapes
        assert self.image.shape[1] == self.cnn_prob.shape[0], f"Image and CNN prob height mismatch: {self.image.shape[1]} vs {self.cnn_prob.shape[0]}"
        assert self.image.shape[2] == self.cnn_prob.shape[1], f"Image and CNN prob width mismatch: {self.image.shape[2]} vs {self.cnn_prob.shape[1]}"
        
        # Extract all possible patch positions
        self.height, self.width = self.cnn_prob.shape
        self.all_positions = []
        
        for i in range(0, self.height - patch_size + 1, patch_size):
            for j in range(0, self.width - patch_size + 1, patch_size):
                self.all_positions.append((i, j))
        
        self.num_patches = len(self.all_positions)
        self.current_patch_idx = 0
        self.current_pos = None
        
        # Action space: continuous threshold adjustment
        self.action_space = spaces.Box(
            low=-0.3,
            high=0.3,
            shape=(1,),
            dtype=np.float32
        )
        
        # Observation space: flattened features
        # Features: CNN prob patch (64x64) + spatial stats (5) + threshold (1)
        obs_dim = patch_size * patch_size + 6
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Track performance
        self.episode_rewards = []
        self.baseline_f1 = None
    
    def _extract_features(self, i, j, threshold):
        """Extract observation features for a patch"""
        # Extract CNN probability patch
        cnn_patch = self.cnn_prob[i:i+self.patch_size, j:j+self.patch_size]
        
        # Spatial statistics
        mean_prob = np.mean(cnn_patch)
        std_prob = np.std(cnn_patch)
        min_prob = np.min(cnn_patch)
        max_prob = np.max(cnn_patch)
        
        # Context: surrounding mean probability
        context_size = 32
        i_start = max(0, i - context_size)
        i_end = min(self.height, i + self.patch_size + context_size)
        j_start = max(0, j - context_size)
        j_end = min(self.width, j + self.patch_size + context_size)
        context_patch = self.cnn_prob[i_start:i_end, j_start:j_end]
        context_mean = np.mean(context_patch)
        
        # Flatten CNN patch and concatenate with stats
        features = np.concatenate([
            cnn_patch.flatten(),
            np.array([mean_prob, std_prob, min_prob, max_prob, context_mean, threshold])
        ])
        
        return features.astype(np.float32)
    
    def _calculate_f1(self, predictions, targets):
        """Calculate F1-score"""
        tp = np.sum((predictions == 1) & (targets == 1))
        fp = np.sum((predictions == 1) & (targets == 0))
        fn = np.sum((predictions == 0) & (targets == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    
    def reset(self, seed=None, options=None):
        """Reset environment and start new episode"""
        super().reset(seed=seed)
        
        # Reset to first patch or random patch
        if options and options.get('random_start', False):
            self.current_patch_idx = np.random.randint(0, self.num_patches)
        else:
            self.current_patch_idx = 0
        
        self.current_pos = self.all_positions[self.current_patch_idx]
        
        # Get initial observation with baseline threshold
        obs = self._extract_features(self.current_pos[0], self.current_pos[1], self.baseline_threshold)
        
        return obs, {}
    
    def step(self, action):
        """Execute threshold adjustment action"""
        i, j = self.current_pos
        
        # Apply threshold adjustment (bounded)
        threshold_delta = np.clip(action[0], -0.3, 0.3)
        adjusted_threshold = np.clip(self.baseline_threshold + threshold_delta, 0.1, 0.9)
        
        # Extract patch
        cnn_patch = self.cnn_prob[i:i+self.patch_size, j:j+self.patch_size]
        gt_patch = self.ground_truth[i:i+self.patch_size, j:j+self.patch_size]
        
        # Apply adjusted threshold
        pred_adjusted = (cnn_patch > adjusted_threshold).astype(np.uint8)
        gt_binary = (gt_patch > 0).astype(np.uint8)
        
        # Calculate F1-score with adjusted threshold
        f1_adjusted = self._calculate_f1(pred_adjusted, gt_binary)
        
        # Calculate baseline F1 (threshold=0.5)
        pred_baseline = (cnn_patch > self.baseline_threshold).astype(np.uint8)
        f1_baseline = self._calculate_f1(pred_baseline, gt_binary)
        
        # Reward: improvement over baseline
        f1_improvement = f1_adjusted - f1_baseline
        
        # Scale reward to encourage exploration
        reward = f1_improvement * 10.0
        
        # Bonus for significant improvements
        if f1_improvement > 0.05:
            reward += 1.0
        
        # Penalty for making things worse
        if f1_improvement < -0.05:
            reward -= 1.0
        
        # Move to next patch
        self.current_patch_idx += 1
        done = (self.current_patch_idx >= self.num_patches)
        
        if not done:
            self.current_pos = self.all_positions[self.current_patch_idx]
            next_obs = self._extract_features(self.current_pos[0], self.current_pos[1], adjusted_threshold)
        else:
            # Episode done
            next_obs = self._extract_features(i, j, adjusted_threshold)  # Dummy observation
        
        info = {
            'f1_adjusted': f1_adjusted,
            'f1_baseline': f1_baseline,
            'f1_improvement': f1_improvement,
            'threshold': adjusted_threshold,
            'threshold_delta': threshold_delta
        }
        
        return next_obs, reward, done, False, info
    
    def render(self):
        """Optional: render current state"""
        pass


# For backwards compatibility with stable-baselines3
class ThresholdRefinementEnvWrapper(ThresholdRefinementEnv):
    """Wrapper to ensure compatibility with stable-baselines3"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
