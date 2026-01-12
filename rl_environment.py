import gymnasium as gym
import numpy as np
from gymnasium import spaces

class CloudMaskRefinementEnv(gym.Env):
    """
    Custom environment for refining cloud masks using RL.
    Each patch is a separate 1-step episode for numerical stability.
    """
    def __init__(self, image, cnn_prob, ground_truth, patch_size=64):
        super().__init__()
        print("🔧 Initializing CloudMaskRefinementEnv - Episode-per-Patch Design")
        self.image = image  # (H, W, bands)
        self.cnn_prob = cnn_prob  # (H, W)
        self.ground_truth = ground_truth  # (H, W) binary
        self.patch_size = patch_size
        self.H, self.W = image.shape[:2]

        # Action space: 0 = not cloud, 1 = cloud
        self.action_space = spaces.Discrete(2)

        # Observation space: patch of image + cnn_prob patch (channels first for PyTorch)
        obs_shape = (image.shape[2] + 1, patch_size, patch_size)  # (C, H, W)
        self.observation_space = spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

        # Generate all patch positions
        self.all_positions = []
        for i in range(0, self.H - patch_size, patch_size):
            for j in range(0, self.W - patch_size, patch_size):
                self.all_positions.append((i, j))
        
        self.position_index = 0
        self.current_pos = self.all_positions[0]
        print(f"📊 Total patches: {len(self.all_positions)}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Get next patch position (cycle through all patches)
        self.current_pos = self.all_positions[self.position_index]
        self.position_index = (self.position_index + 1) % len(self.all_positions)
        return self._get_obs(), {}

    def _get_obs(self):
        i, j = self.current_pos
        patch_image = self.image[i:i+self.patch_size, j:j+self.patch_size]
        patch_prob = self.cnn_prob[i:i+self.patch_size, j:j+self.patch_size]
        patch_prob = patch_prob[..., np.newaxis]  # (patch, patch, 1)
        obs = np.concatenate([patch_image, patch_prob], axis=-1)  # (H, W, C)
        obs = np.transpose(obs, (2, 0, 1))  # Convert to (C, H, W) for PyTorch
        return obs.astype(np.float32)

    def step(self, action):
        i, j = self.current_pos
        patch_gt = self.ground_truth[i:i+self.patch_size, j:j+self.patch_size]

        total_pixels = patch_gt.size
        cloud_pixels = np.sum(patch_gt == 1)
        cloud_ratio = cloud_pixels / total_pixels

        # Reward structure optimized for low-coverage cloud detection
        if cloud_pixels == 0:
            # Pure clear sky (rare in this dataset)
            reward = 3.0 if action == 0 else -5.0
        elif cloud_ratio < 0.05:
            # Very few clouds (< 5%) - still important to detect
            if action == 1:
                reward = 2.0 + 3.0 * cloud_ratio  # Base reward + bonus
            else:
                reward = -4.0  # Fixed penalty for missing any clouds
        else:
            # Any significant clouds (5%+) - prioritize detection
            if action == 1:
                reward = 3.0 + 2.0 * cloud_ratio  # Good reward for detection
            else:
                reward = -6.0 * cloud_ratio  # Heavy penalty for missing clouds

        # Episode ALWAYS ends after one step
        done = True
        
        info = {
            'patch_position': (i, j),
            'cloud_pixels': cloud_pixels,
            'action': action
        }

        # Next observation doesn't matter since episode is done
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs, reward, done, False, info

    def render(self):
        pass