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

        # Observation space: patch of image + cnn_prob patch (channels first for PyTorch)
        obs_shape = (image.shape[2] + 1, patch_size, patch_size)  # (C, H, W)
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
        obs = np.concatenate([patch_image, patch_prob], axis=-1)  # (H, W, C)
        obs = np.transpose(obs, (2, 0, 1))  # Convert to (C, H, W) for PyTorch
        return obs.astype(np.float32)

    def step(self, action):
        i, j = self.current_pos
        # Get the entire patch ground truth
        patch_gt = self.ground_truth[i:i+self.patch_size, j:j+self.patch_size]

        # Calculate rewards based on patch-level performance
        patch_pred = np.full_like(patch_gt, action)  # Predict same action for entire patch

        # Calculate true positives, false positives, false negatives
        tp = np.sum((patch_pred == 1) & (patch_gt == 1))  # Correctly predicted clouds
        fp = np.sum((patch_pred == 1) & (patch_gt == 0))  # Incorrectly predicted clouds
        fn = np.sum((patch_pred == 0) & (patch_gt == 1))  # Missed clouds
        tn = np.sum((patch_pred == 0) & (patch_gt == 0))  # Correctly predicted clear

        total_pixels = patch_gt.size
        cloud_pixels = np.sum(patch_gt == 1)
        clear_pixels = np.sum(patch_gt == 0)

        # Balanced reward structure
        if cloud_pixels == 0:
            # No clouds in patch - reward for correct clear prediction
            reward = 1.0 if action == 0 else -0.5
        else:
            # Clouds present - balance precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / cloud_pixels if cloud_pixels > 0 else 0

            # Reward = weighted combination of precision and recall with stronger precision emphasis
            reward = 0.7 * precision + 0.3 * recall

            # Penalties for false positives and missed clouds
            if action == 1 and fp > 0:  # Predicted clouds incorrectly
                reward -= 0.2 * (fp / total_pixels)  # Penalty proportional to false positive rate
            if action == 0 and fn > 0:  # Missed clouds
                reward -= 0.4 * (fn / cloud_pixels)  # Stronger penalty for missing clouds

        # Move to next patch (simple grid traversal)
        j += self.patch_size
        if j >= self.W - self.patch_size:
            j = 0
            i += self.patch_size
            if i >= self.H - self.patch_size:
                self.done = True

        self.current_pos = (i, j)
        obs = self._get_obs() if not self.done else np.zeros(self.observation_space.shape)

        # Return patch position in info for evaluation
        info = {'patch_position': (i, j)} if not self.done else {}

        return obs, reward, self.done, info

    def render(self):
        pass