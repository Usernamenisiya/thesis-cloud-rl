import gymnasium as gym
import numpy as np
from gymnasium import spaces

class CloudMaskRefinementEnv(gym.Env):
    """
    Custom environment for refining cloud masks using RL.
    AGGRESSIVE reward structure to force cloud detection.
    """
    def __init__(self, image, cnn_prob, ground_truth, patch_size=64):
        super().__init__()
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

        self.current_pos = (0, 0)
        self.done = False
        self.episode_clouds_detected = 0
        self.episode_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_pos = (0, 0)
        self.done = False
        self.episode_clouds_detected = 0
        self.episode_steps = 0
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

        # BALANCED REWARD STRUCTURE - Optimize F1-Score
        print(f"DEBUG: Using BALANCED rewards - cloud_pixels={cloud_pixels}, action={action}")  # DEBUG
        if cloud_pixels == 0:
            # Pure clear sky patch
            if action == 0:
                reward = 0.1  # Small reward for correct clear
            else:
                reward = -0.8  # Moderate penalty for false positive
        elif cloud_pixels >= total_pixels * 0.3:
            # Significant clouds present - prioritize detection but penalize over-prediction
            if action == 1:
                reward = 0.8 + (cloud_pixels / total_pixels)  # Good reward for correct detection
                self.episode_clouds_detected += 1
            else:
                reward = -1.2 * (cloud_pixels / total_pixels)  # Moderate penalty for missing clouds
        else:
            # Few clouds - balance precision and recall
            if action == 1:
                reward = 0.3 * (cloud_pixels / total_pixels)  # Smaller reward for partial clouds
                self.episode_clouds_detected += 1
            else:
                reward = -0.4 * (cloud_pixels / total_pixels)  # Smaller penalty for missing few clouds

        self.episode_steps += 1

        # Move to next patch
        j += self.patch_size
        if j >= self.W - self.patch_size:
            j = 0
            i += self.patch_size
            if i >= self.H - self.patch_size:
                self.done = True

        self.current_pos = (i, j)
        obs = self._get_obs() if not self.done else np.zeros(self.observation_space.shape)

        # Episode-end penalty if agent detected ZERO clouds
        if self.done and self.episode_clouds_detected == 0:
            reward -= 1.0

        info = {'patch_position': (i, j)} if not self.done else {
            'clouds_detected': self.episode_clouds_detected,
            'episode_steps': self.episode_steps
        }

        return obs, reward, self.done, False, info

    def render(self):
        pass