import gym
import numpy as np
from gym import spaces

class CloudMaskRefinementEnv(gym.Env):
    """
    Custom environment for refining cloud masks using RL.
    The agent observes a patch of the image and CNN probability, and decides if it's cloud.
    """
    def __init__(self, image, cnn_prob, ground_truth, patch_size=32):
        super(CloudMaskRefinementEnv, self).__init__()
        self.image = image  # (H, W, bands)
        self.cnn_prob = cnn_prob  # (H, W)
        self.ground_truth = ground_truth  # (H, W) binary
        self.patch_size = patch_size
        self.H, self.W = image.shape[:2]

        # Action space: 0 = not cloud, 1 = cloud
        self.action_space = spaces.Discrete(2)

        # Observation space: patch of image + cnn_prob patch
        obs_shape = (patch_size, patch_size, image.shape[2] + 1)  # image bands + prob
        self.observation_space = spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

        self.current_pos = (0, 0)
        self.done = False

    def reset(self):
        self.current_pos = (0, 0)
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        i, j = self.current_pos
        patch_image = self.image[i:i+self.patch_size, j:j+self.patch_size]
        patch_prob = self.cnn_prob[i:i+self.patch_size, j:j+self.patch_size]
        patch_prob = patch_prob[..., np.newaxis]  # (patch, patch, 1)
        obs = np.concatenate([patch_image, patch_prob], axis=-1)
        return obs.astype(np.float32)

    def step(self, action):
        i, j = self.current_pos
        true_label = self.ground_truth[i + self.patch_size//2, j + self.patch_size//2]  # center pixel
        reward = 1 if action == true_label else -1

        # Move to next patch (simple grid traversal)
        j += self.patch_size
        if j >= self.W - self.patch_size:
            j = 0
            i += self.patch_size
            if i >= self.H - self.patch_size:
                self.done = True

        self.current_pos = (i, j)
        obs = self._get_obs() if not self.done else np.zeros(self.observation_space.shape)
        return obs, reward, self.done, {}

    def render(self):
        pass