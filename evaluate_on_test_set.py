"""
Evaluate saved PPO model on the 20-patch test set
"""
import numpy as np
import rasterio
from stable_baselines3 import PPO
from rl_environment import CloudMaskRefinementEnv
from s2cloudless import S2PixelCloudDetector
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import glob
import os

def load_sentinel2_image(image_path):
    """Load and normalize Sentinel-2 image"""
    with rasterio.open(image_path) as src:
        image = src.read()
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image / 10000.0, 0, 1)
    return image.astype(np.float32)

def get_cloud_mask(image):
    """Generate CNN cloud probability mask"""
    detector = S2PixelCloudDetector(threshold=0.5, all_bands=True, average_over=1, dilation_size=0)
    image_int = (image * 10000).astype(np.int16)
    image_reshaped = np.transpose(image_int, (2, 0, 1))[np.newaxis, ...]
    cloud_prob = detector.get_cloud_probability_maps(image_reshaped)
    return cloud_prob[0].astype(np.float32)

if __name__ == "__main__":
    print("="*60)
    print("📊 Evaluating PPO Model on Test Set")
    print("="*60)
    
    # Load test data (20 patches: indices 80-100)
    data_dir = "data/cloudsen12_processed"
    image_files = sorted(glob.glob(os.path.join(data_dir, "image_*.tif")))
    mask_files = sorted(glob.glob(os.path.join(data_dir, "mask_*.tif")))
    
    # 80/20 split
    split_idx = int(0.8 * len(image_files))
    test_image_files = image_files[split_idx:]
    test_mask_files = mask_files[split_idx:]
    
    print(f"📁 Found {len(test_image_files)} test patches")
    
    # Load saved model
    model_path = "ppo_cloud_refinement_model_final/policy.pth"
    print(f"\n🤖 Loading model from: {model_path}")
    model = PPO.load(model_path.replace("/policy.pth", ""))
    print("✅ Model loaded successfully")
    
    # Evaluate on all test patches
    print(f"\n📊 Evaluating on {len(test_image_files)} test patches...")
    
    all_gt = []
    all_cnn = []
    all_ppo = []
    
    for idx, (img_path, mask_path) in enumerate(zip(test_image_files, test_mask_files)):
        print(f"  Processing test patch {idx+1}/{len(test_image_files)}", end='\r')
        
        # Load test patch
        test_image = load_sentinel2_image(img_path)
        test_cnn_prob = get_cloud_mask(test_image)
        
        with rasterio.open(mask_path) as src:
            test_gt = src.read(1)
        
        # Create evaluation environment for this patch
        eval_env = CloudMaskRefinementEnv(test_image, test_cnn_prob, test_gt, patch_size=64)
        rl_predictions = np.zeros_like(test_gt, dtype=np.uint8)
        
        # Evaluate all patches (each is a separate episode)
        num_patches = len(eval_env.all_positions)
        
        for patch_idx in range(num_patches):
            obs, _ = eval_env.reset()
            i, j = eval_env.current_pos
            patch_size = eval_env.patch_size
            
            action, _ = model.predict(obs, deterministic=True)
            rl_predictions[i:i+patch_size, j:j+patch_size] = action
            
            obs, reward, done, truncated, info = eval_env.step(action)
        
        # Collect predictions
        gt_binary = (test_gt > 0).astype(np.uint8)
        cnn_binary = (test_cnn_prob > 0.5).astype(np.uint8)
        rl_binary = (rl_predictions > 0).astype(np.uint8)
        
        all_gt.append(gt_binary.flatten())
        all_cnn.append(cnn_binary.flatten())
        all_ppo.append(rl_binary.flatten())
    
    print(f"\n✅ Evaluation completed on {len(test_image_files)} test patches")
    
    # Combine all test patches
    all_gt = np.concatenate(all_gt)
    all_cnn = np.concatenate(all_cnn)
    all_ppo = np.concatenate(all_ppo)
    
    # CNN metrics on test set
    cnn_accuracy = accuracy_score(all_gt, all_cnn)
    cnn_precision = precision_score(all_gt, all_cnn, zero_division=0)
    cnn_recall = recall_score(all_gt, all_cnn, zero_division=0)
    cnn_f1 = f1_score(all_gt, all_cnn, zero_division=0)
    
    # PPO metrics on test set
    ppo_accuracy = accuracy_score(all_gt, all_ppo)
    ppo_precision = precision_score(all_gt, all_ppo, zero_division=0)
    ppo_recall = recall_score(all_gt, all_ppo, zero_division=0)
    ppo_f1 = f1_score(all_gt, all_ppo, zero_division=0)
    
    # Calculate improvements
    f1_improvement = ((ppo_f1 - cnn_f1) / cnn_f1 * 100) if cnn_f1 > 0 else 0
    accuracy_improvement = ((ppo_accuracy - cnn_accuracy) / cnn_accuracy * 100) if cnn_accuracy > 0 else 0
    
    print("\n" + "=" * 60)
    print(f"📈 TEST SET RESULTS ({len(test_image_files)} patches, {len(all_gt):,} pixels)")
    print("=" * 60)
    
    print("\n🧠 CNN Baseline:")
    print(f"  Accuracy:  {cnn_accuracy:.4f} ({cnn_accuracy*100:.2f}%)")
    print(f"  Precision: {cnn_precision:.4f}")
    print(f"  Recall:    {cnn_recall:.4f}")
    print(f"  F1-Score:  {cnn_f1:.4f}")
    
    print("\n🤖 PPO Refined:")
    print(f"  Accuracy:  {ppo_accuracy:.4f} ({ppo_accuracy*100:.2f}%)")
    print(f"  Precision: {ppo_precision:.4f}")
    print(f"  Recall:    {ppo_recall:.4f}")
    print(f"  F1-Score:  {ppo_f1:.4f}")
    
    print("\n🎯 Improvements:")
    print(f"  F1-Score:  {f1_improvement:+.2f}%")
    print(f"  Accuracy:  {accuracy_improvement:+.2f}%")
    print(f"  Precision: {ppo_precision - cnn_precision:+.4f}")
    print(f"  Recall:    {ppo_recall - cnn_recall:+.4f}")
    
    print("\n" + "=" * 60)
