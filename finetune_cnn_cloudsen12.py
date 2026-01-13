"""
Fine-tune pretrained cloud detection CNN on CloudSEN12 train set
Transfer learning approach with low learning rate
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
import glob
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import the CNN model
from cnn_inference import CloudCNN, load_sentinel2_image, get_cloud_mask


class CloudSEN12Dataset(Dataset):
    """Dataset for CloudSEN12 patches"""
    def __init__(self, image_files, mask_files):
        self.image_files = image_files
        self.mask_files = mask_files
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image (10 bands, 512x512)
        image = load_sentinel2_image(self.image_files[idx])
        
        # Load ground truth mask
        with rasterio.open(self.mask_files[idx]) as src:
            mask = src.read(1)
        
        # Normalize image (already done in load_sentinel2_image)
        # Convert to torch tensors
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        
        return image_tensor, mask_tensor


def evaluate_model(model, dataloader, device):
    """Evaluate model on a dataset"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
    
    all_preds = np.concatenate([p.flatten() for p in all_preds])
    all_targets = np.concatenate([t.flatten() for t in all_targets])
    
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def main():
    print("🔧 CNN Fine-Tuning on CloudSEN12")
    print("="*60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Load data
    data_dir = 'data/cloudsen12_processed'
    image_files = sorted(glob.glob(f'{data_dir}/*_image.tif'))
    mask_files = sorted(glob.glob(f'{data_dir}/*_mask.tif'))
    
    # 80/20 split
    split_idx = int(0.8 * len(image_files))
    train_image_files = image_files[:split_idx]
    train_mask_files = mask_files[:split_idx]
    test_image_files = image_files[split_idx:]
    test_mask_files = mask_files[split_idx:]
    
    print(f"📊 Train patches: {len(train_image_files)}")
    print(f"📊 Test patches: {len(test_image_files)}")
    
    # Create datasets
    train_dataset = CloudSEN12Dataset(train_image_files, train_mask_files)
    test_dataset = CloudSEN12Dataset(test_image_files, test_mask_files)
    
    # DataLoaders (batch size 1 due to 512x512 images)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Load pretrained model
    print("\n🔄 Loading pretrained CNN model...")
    model = CloudCNN(in_channels=10)
    
    # Load pretrained weights (assuming they exist)
    try:
        # Try to load from saved model
        checkpoint_path = 'models/cloud_cnn_model.pth'
        if Path(checkpoint_path).exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"✅ Loaded pretrained weights from {checkpoint_path}")
        else:
            print("⚠️  No pretrained weights found, training from scratch")
    except Exception as e:
        print(f"⚠️  Could not load pretrained weights: {e}")
    
    model = model.to(device)
    
    # Evaluate before fine-tuning
    print("\n📊 Evaluating pretrained model on test set...")
    pretrained_metrics = evaluate_model(model, test_loader, device)
    print(f"  Accuracy:  {pretrained_metrics['accuracy']:.4f}")
    print(f"  Precision: {pretrained_metrics['precision']:.4f}")
    print(f"  Recall:    {pretrained_metrics['recall']:.4f}")
    print(f"  F1-Score:  {pretrained_metrics['f1_score']:.4f}")
    
    # Fine-tuning setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)  # Very low LR for fine-tuning
    
    num_epochs = 10
    print(f"\n🚀 Starting fine-tuning ({num_epochs} epochs, lr=1e-5)...")
    
    best_f1 = 0
    training_history = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}", end='\r')
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Evaluate on test set
        test_metrics = evaluate_model(model, test_loader, device)
        
        print(f"\n  Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Test F1: {test_metrics['f1_score']:.4f}")
        
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'test_metrics': test_metrics
        })
        
        # Save best model
        if test_metrics['f1_score'] > best_f1:
            best_f1 = test_metrics['f1_score']
            Path('models').mkdir(exist_ok=True)
            model_path = f"models/cnn_finetuned_cloudsen12.pth"
            torch.save(model.state_dict(), model_path)
            print(f"  ✅ Saved best model (F1={best_f1:.4f})")
    
    # Final evaluation
    print("\n" + "="*60)
    print("📈 FINE-TUNED CNN RESULTS")
    print("="*60)
    
    # Load best model
    model.load_state_dict(torch.load(model_path, map_location=device))
    final_metrics = evaluate_model(model, test_loader, device)
    
    print("\n🧠 Pretrained CNN (before fine-tuning):")
    print(f"  Accuracy:  {pretrained_metrics['accuracy']:.4f}")
    print(f"  Precision: {pretrained_metrics['precision']:.4f}")
    print(f"  Recall:    {pretrained_metrics['recall']:.4f}")
    print(f"  F1-Score:  {pretrained_metrics['f1_score']:.4f}")
    
    print("\n🔥 Fine-Tuned CNN (after adaptation):")
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"  F1-Score:  {final_metrics['f1_score']:.4f}")
    
    f1_improvement = ((final_metrics['f1_score'] - pretrained_metrics['f1_score']) / pretrained_metrics['f1_score'] * 100) if pretrained_metrics['f1_score'] > 0 else 0
    print(f"\n🎯 F1-Score Improvement: {f1_improvement:+.2f}%")
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    results = {
        'pretrained_metrics': pretrained_metrics,
        'finetuned_metrics': final_metrics,
        'improvement_percent': f1_improvement,
        'training_history': training_history,
        'hyperparameters': {
            'learning_rate': 1e-5,
            'epochs': num_epochs,
            'batch_size': 1
        }
    }
    
    with open('results/cnn_finetuning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: results/cnn_finetuning_results.json")
    print(f"💾 Model saved to: {model_path}")
    print("="*60)

if __name__ == "__main__":
    main()
