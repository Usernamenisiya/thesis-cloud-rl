import gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from rl_environment import CloudMaskRefinementEnv
import numpy as np
from cnn_inference import load_sentinel2_image, get_cloud_mask
import rasterio

def load_ground_truth(gt_path):
    """
    Load ground truth mask.
    Assume binary .tif or .png.
    """
    with rasterio.open(gt_path) as src:
        gt = src.read(1).astype(np.uint8)  # assume single band
    return gt

# Load data
image_path = "data/sentinel2_image.tif"  # Updated path
gt_path = "data/ground_truth.tif"  # Updated path

image = load_sentinel2_image(image_path)
cnn_prob = get_cloud_mask(image)
ground_truth = load_ground_truth(gt_path)

# Ensure shapes match
assert image.shape[:2] == ground_truth.shape, "Image and ground truth shapes must match"

# Create environment
env = CloudMaskRefinementEnv(image, cnn_prob, ground_truth)

# Train DQN
model = DQN("CnnPolicy", env, verbose=1, buffer_size=10000, learning_starts=1000)
model.learn(total_timesteps=10000)

# Save model
model.save("dqn_cloud_refinement")

# To evaluate
obs = env.reset()
refined_mask = np.zeros_like(ground_truth)
while not env.done:
    action, _ = model.predict(obs)
    # Assuming action corresponds to center pixel
    i, j = env.current_pos
    center_i = i + env.patch_size // 2
    center_j = j + env.patch_size // 2
    refined_mask[center_i, center_j] = action
    obs, reward, done, info = env.step(action)

print("Refined mask shape:", refined_mask.shape)