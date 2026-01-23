import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json

# Colab detection
IN_COLAB = ('google.colab' in sys.modules or 
            os.path.exists('/content') or 
            'COLAB_GPU' in os.environ)

if IN_COLAB:
    # Drive should already be mounted from notebook cells
    BASE_DIR = Path('/content/drive/MyDrive/Colab_Data')
else:
    BASE_DIR = Path('.')

# Import required modules
from stable_baselines3 import PPO, DQN
from rl_thin_cloud_environment import ThinCloudDetectionEnv
from rl_thin_cloud_environment_discrete import ThinCloudDetectionEnvDiscrete


def load_test_data():
    """Load test patches from TIF files."""
    print("Loading test data...")
    
    import glob
    import rasterio
    
    # Load from processed TIF files on Drive
    data_dir = '/content/drive/MyDrive/Colab_Data/cloudsen12_processed_1000'
    image_files = sorted(glob.glob(f'{data_dir}/*_image.tif'))
    mask_files = sorted(glob.glob(f'{data_dir}/*_mask.tif'))
    
    # Use test set (last 200 patches, indices 800-999)
    split_idx = int(0.8 * len(image_files))
    test_images = image_files[split_idx:]
    test_masks = mask_files[split_idx:]
    
    print(f"Found {len(test_images)} test images")
    
    patches = []
    labels = []
    
    for img_path, mask_path in zip(test_images, test_masks):
        # Load image
        with rasterio.open(img_path) as src:
            patch = src.read()  # (bands, H, W)
            patch = np.transpose(patch, (1, 2, 0))  # (H, W, bands)
        
        # Load mask
        with rasterio.open(mask_path) as src:
            label = src.read(1)  # (H, W)
        
        patches.append(patch)
        labels.append(label)
    
    return np.array(patches), np.array(labels)


def apply_cnn_baseline(patch):
    """Apply CNN baseline threshold detection."""
    # Normalize bands
    bands_mean = np.array([1353.3418, 1265.7446, 1269.3455, 1404.6283, 
                           2033.6624, 2583.3354, 2743.1646, 2895.0786, 
                           2927.9084, 747.4245, 16.481863, 2383.0974])
    bands_std = np.array([72.612915, 156.40573, 240.63678, 373.3985, 
                          563.72205, 722.8531, 846.32776, 975.0849, 
                          1084.3259, 314.12592, 21.04352, 833.8011])
    
    normalized_patch = (patch - bands_mean) / bands_std
    
    # CNN-like thresholds based on spectral features
    b3 = normalized_patch[:, :, 2]  # Green
    b8 = normalized_patch[:, :, 7]  # NIR
    b11 = normalized_patch[:, :, 10]  # SWIR
    
    # Cloud detection threshold
    cloud_score = (b3 + b8 - b11) / 3.0
    cloud_mask = (cloud_score > 0.15).astype(np.uint8)
    
    return cloud_mask


def apply_ppo_model(patch, model):
    """Apply PPO model to get cloud mask."""
    env = ThinCloudDetectionEnv(patch)
    obs, _ = env.reset()
    
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    
    return env.current_prediction


def apply_dqn_model(patch, model):
    """Apply DQN model to get cloud mask."""
    env = ThinCloudDetectionEnvDiscrete(patch)
    obs, _ = env.reset()
    
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    
    return env.current_prediction


def create_rgb_image(patch):
    """Create RGB image from patch for visualization."""
    # Use bands 3, 2, 1 (RGB)
    rgb = patch[:, :, [2, 1, 0]]
    
    # Normalize to 0-1 range for display
    rgb = np.clip(rgb / 3000.0, 0, 1)
    
    # Apply gamma correction for better visualization
    rgb = np.power(rgb, 0.7)
    
    return rgb


def create_false_color(patch):
    """Create false color composite (NIR, Red, Green)."""
    # Use bands 7 (NIR), 3 (Red), 2 (Green)
    false_color = patch[:, :, [7, 3, 2]]
    
    # Normalize
    false_color = np.clip(false_color / 3000.0, 0, 1)
    false_color = np.power(false_color, 0.7)
    
    return false_color


def visualize_comparison(patches, labels, cnn_masks, ppo_masks, dqn_masks, 
                        num_samples=5, save_path=None):
    """Create side-by-side comparison visualization."""
    
    fig, axes = plt.subplots(num_samples, 6, figsize=(24, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # RGB Image
        rgb = create_rgb_image(patches[i])
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title('RGB Image' if i == 0 else '', fontsize=12)
        axes[i, 0].axis('off')
        
        # False Color
        false_color = create_false_color(patches[i])
        axes[i, 1].imshow(false_color)
        axes[i, 1].set_title('False Color (NIR-R-G)' if i == 0 else '', fontsize=12)
        axes[i, 1].axis('off')
        
        # Ground Truth
        gt_display = np.zeros((*labels[i].shape, 3))
        gt_display[labels[i] == 0] = [0.2, 0.2, 0.2]  # Clear - dark gray
        gt_display[labels[i] == 1] = [1.0, 1.0, 1.0]  # Thick cloud - white
        gt_display[labels[i] == 2] = [0.7, 0.9, 1.0]  # Thin cloud - light blue
        gt_display[labels[i] == 3] = [0.4, 0.2, 0.0]  # Shadow - brown
        
        axes[i, 2].imshow(gt_display)
        axes[i, 2].set_title('Ground Truth' if i == 0 else '', fontsize=12)
        axes[i, 2].axis('off')
        
        # CNN Baseline
        cnn_display = np.zeros((*cnn_masks[i].shape, 3))
        cnn_display[cnn_masks[i] == 0] = [0.2, 0.2, 0.2]
        cnn_display[cnn_masks[i] == 1] = [1.0, 1.0, 1.0]
        
        axes[i, 3].imshow(cnn_display)
        axes[i, 3].set_title('CNN Baseline' if i == 0 else '', fontsize=12)
        axes[i, 3].axis('off')
        
        # PPO
        ppo_display = np.zeros((*ppo_masks[i].shape, 3))
        ppo_display[ppo_masks[i] == 0] = [0.2, 0.2, 0.2]
        ppo_display[ppo_masks[i] == 1] = [1.0, 1.0, 1.0]
        
        axes[i, 4].imshow(ppo_display)
        axes[i, 4].set_title('PPO' if i == 0 else '', fontsize=12)
        axes[i, 4].axis('off')
        
        # DQN
        dqn_display = np.zeros((*dqn_masks[i].shape, 3))
        dqn_display[dqn_masks[i] == 0] = [0.2, 0.2, 0.2]
        dqn_display[dqn_masks[i] == 1] = [1.0, 1.0, 1.0]
        
        axes[i, 5].imshow(dqn_display)
        axes[i, 5].set_title('DQN' if i == 0 else '', fontsize=12)
        axes[i, 5].axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=[0.2, 0.2, 0.2], label='Clear/No Cloud'),
        mpatches.Patch(color=[1.0, 1.0, 1.0], label='Cloud (Thick or Thin)'),
        mpatches.Patch(color=[0.7, 0.9, 1.0], label='Thin Cloud (GT only)'),
        mpatches.Patch(color=[0.4, 0.2, 0.0], label='Shadow (GT only)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=11)
    
    plt.tight_layout(rect=[0, 0.01, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def main():
    print("="*60)
    print("Algorithm Mask Comparison Visualization")
    print("="*60)
    
    # Load test data
    patches, labels = load_test_data()
    print(f"Loaded {len(patches)} test patches")
    
    # Load PPO model
    print("\nLoading PPO model...")
    ppo_model_path = BASE_DIR / 'thin_cloud_v2' / 'thin_cloud_720000_steps.zip'
    if not ppo_model_path.exists():
        print(f"ERROR: PPO model not found at {ppo_model_path}")
        return
    ppo_model = PPO.load(str(ppo_model_path))
    print("PPO model loaded successfully")
    
    # Load DQN model
    print("Loading DQN model...")
    dqn_model_path = BASE_DIR / 'dqn_thin_cloud' / 'dqn_thin_cloud_100000_steps.zip'
    if not dqn_model_path.exists():
        print(f"ERROR: DQN model not found at {dqn_model_path}")
        return
    dqn_model = DQN.load(str(dqn_model_path))
    print("DQN model loaded successfully")
    
    # Select diverse test samples
    # Choose patches with varying cloud coverage
    print("\nSelecting diverse test samples...")
    
    # Calculate cloud coverage for each patch
    thin_cloud_ratios = []
    for label in labels:
        thin_ratio = np.sum(label == 2) / label.size
        thin_cloud_ratios.append(thin_ratio)
    
    thin_cloud_ratios = np.array(thin_cloud_ratios)
    
    # Select 5 samples with different thin cloud coverage
    selected_indices = []
    
    # 1. High thin cloud coverage
    high_thin_idx = np.argsort(thin_cloud_ratios)[-1]
    selected_indices.append(high_thin_idx)
    
    # 2. Medium-high thin cloud
    med_high_idx = np.argsort(thin_cloud_ratios)[-len(thin_cloud_ratios)//4]
    selected_indices.append(med_high_idx)
    
    # 3. Medium thin cloud
    med_idx = np.argsort(thin_cloud_ratios)[len(thin_cloud_ratios)//2]
    selected_indices.append(med_idx)
    
    # 4. Low thin cloud
    low_idx = np.argsort(thin_cloud_ratios)[len(thin_cloud_ratios)//4]
    selected_indices.append(low_idx)
    
    # 5. Very low/no thin cloud
    very_low_idx = np.argsort(thin_cloud_ratios)[0]
    selected_indices.append(very_low_idx)
    
    selected_patches = [patches[i] for i in selected_indices]
    selected_labels = [labels[i] for i in selected_indices]
    
    print(f"Selected patch indices: {selected_indices}")
    print(f"Thin cloud ratios: {[thin_cloud_ratios[i] for i in selected_indices]}")
    
    # Generate predictions for all three algorithms
    print("\nGenerating predictions...")
    
    cnn_masks = []
    ppo_masks = []
    dqn_masks = []
    
    for i, patch in enumerate(selected_patches):
        print(f"  Processing patch {i+1}/5...")
        
        # CNN baseline
        cnn_mask = apply_cnn_baseline(patch)
        cnn_masks.append(cnn_mask)
        
        # PPO
        ppo_mask = apply_ppo_model(patch, ppo_model)
        ppo_masks.append(ppo_mask)
        
        # DQN
        dqn_mask = apply_dqn_model(patch, dqn_model)
        dqn_masks.append(dqn_mask)
    
    print("All predictions generated")
    
    # Create visualization
    print("\nCreating visualization...")
    save_path = None
    if IN_COLAB:
        save_path = BASE_DIR / 'algorithm_comparison' / 'algorithm_mask_comparison.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        save_path = Path('results') / 'algorithm_mask_comparison.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    visualize_comparison(
        selected_patches, 
        selected_labels, 
        cnn_masks, 
        ppo_masks, 
        dqn_masks,
        num_samples=5,
        save_path=str(save_path)
    )
    
    # Calculate and display metrics for selected patches
    print("\n" + "="*60)
    print("Metrics for Selected Patches:")
    print("="*60)
    
    for i, (label, cnn, ppo, dqn) in enumerate(zip(selected_labels, cnn_masks, ppo_masks, dqn_masks)):
        print(f"\nPatch {i+1} (Index {selected_indices[i]}):")
        
        # Convert labels to binary (cloud vs no cloud)
        gt_binary = (label > 0).astype(int)
        thin_cloud_mask = (label == 2)
        
        # Calculate thin cloud IoU for each method
        if np.sum(thin_cloud_mask) > 0:
            cnn_thin_iou = np.sum((cnn == 1) & thin_cloud_mask) / np.sum((cnn == 1) | thin_cloud_mask)
            ppo_thin_iou = np.sum((ppo == 1) & thin_cloud_mask) / np.sum((ppo == 1) | thin_cloud_mask)
            dqn_thin_iou = np.sum((dqn == 1) & thin_cloud_mask) / np.sum((dqn == 1) | thin_cloud_mask)
            
            print(f"  Thin Cloud Coverage: {thin_cloud_ratios[selected_indices[i]]*100:.1f}%")
            print(f"  Thin Cloud IoU:")
            print(f"    CNN: {cnn_thin_iou*100:.2f}%")
            print(f"    PPO: {ppo_thin_iou*100:.2f}%")
            print(f"    DQN: {dqn_thin_iou*100:.2f}%")
        else:
            print(f"  No thin clouds in this patch")
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == '__main__':
    main()
